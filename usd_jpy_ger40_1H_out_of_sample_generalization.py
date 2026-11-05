import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
FIXED_RISK_USD = 50.0 
COMMISSION_PER_LOT = 6.0 
TIMEFRAME = 60 # 1 ЧАС

# Конфигурация для разных инструментов
INSTRUMENTS = {
    "USDJPY": {"pip_size": 0.01, "wrn_step": 0.50, "pip_value_usd": 10.0},
    "GER40":  {"pip_size": 1.0,  "wrn_step": 500.0, "pip_value_usd": 1.0} 
    # GER40: 1 пипс = 1 пункт, стоимость на лот ~$1 (зависит от брокера)
}

def run_research(SYMBOL, config):
    print(f"\n{'='*60}")
    print(f"=== ИССЛЕДОВАНИЕ: {SYMBOL} на {TIMEFRAME}M ===")
    print(f"{'='*60}")
    
    pip_size = config['pip_size']
    wrn_step = config['wrn_step']
    pip_val = config['pip_value_usd']
    
    base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
    time_offset_sql = "INTERVAL '1 hour'" 
    con = duckdb.connect(':memory:')

    # SQL: Свип + Engulfing + WRN + Сессии (без жесткого фильтра)
    query_dataset = f"""
    WITH raw_ticks AS (
        SELECT timestamp, bid, ask, volume FROM read_parquet('{base_path}')
    ),
    base_data AS (
        SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time,
            arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
            avg(ask - bid) AS avg_spread
        FROM raw_ticks GROUP BY bar_time
    ),
    with_atr AS (
        SELECT *, GREATEST(high - low, ABS(high - LAG(close) OVER (ORDER BY bar_time)), ABS(low - LAG(close) OVER (ORDER BY bar_time))) AS tr
        FROM base_data
    ),
    with_avg_atr AS (
        SELECT *, AVG(tr) OVER (ORDER BY bar_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS atr
        FROM with_atr
    ),
    -- Диапазоны сессий
    session_ranges AS (
        SELECT 
            date_trunc('day', bar_time + {time_offset_sql}) AS adj_day,
            max(CASE WHEN EXTRACT(hour FROM (bar_time + {time_offset_sql})) >= 3 AND EXTRACT(hour FROM (bar_time + {time_offset_sql})) < 11 THEN high END) AS asian_high,
            min(CASE WHEN EXTRACT(hour FROM (bar_time + {time_offset_sql})) >= 3 AND EXTRACT(hour FROM (bar_time + {time_offset_sql})) < 11 THEN low END) AS asian_low,
            max(CASE WHEN EXTRACT(hour FROM (bar_time + {time_offset_sql})) >= 12 AND EXTRACT(hour FROM (bar_time + {time_offset_sql})) < 16 THEN high END) AS london_high,
            min(CASE WHEN EXTRACT(hour FROM (bar_time + {time_offset_sql})) >= 12 AND EXTRACT(hour FROM (bar_time + {time_offset_sql})) < 16 THEN low END) AS london_low
        FROM with_avg_atr
        GROUP BY adj_day
    ),
    with_prev AS (
        SELECT *, 
            LAG(open) OVER (ORDER BY bar_time) AS open_1, LAG(close) OVER (ORDER BY bar_time) AS close_1,
            LAG(high) OVER (ORDER BY bar_time) AS high_1, LAG(low) OVER (ORDER BY bar_time) AS low_1
        FROM with_avg_atr
    ),
    weekly_range AS (
        SELECT date_trunc('week', bar_time) AS week_start, max(high) AS prev_high, min(low) AS prev_low FROM with_avg_atr GROUP BY week_start
    ),
    wrn_levels AS (
        SELECT b.bar_time, CAST(t.level_int AS DOUBLE) * {wrn_step} AS wrn_level
        FROM with_avg_atr b
        JOIN weekly_range w ON date_trunc('week', b.bar_time) = w.week_start + INTERVAL '7 days'
        CROSS JOIN UNNEST(generate_series(CAST(floor(w.prev_low / {wrn_step}) AS BIGINT) - 1, CAST(ceil(w.prev_high / {wrn_step}) AS BIGINT) + 1, 1)) AS t(level_int)
    ),
    raw_signals AS (
        SELECT 
            p.bar_time AS signal_time, p.close AS entry_price, p.avg_spread, p.low, p.high, p.atr,
            s.asian_high, s.asian_low, s.london_high, s.london_low,
            
            -- MoE
            CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS moe_bull_engulf,
            CASE WHEN (p.close_1 > p.open_1) AND (p.close < p.open) AND (p.close < p.open_1) THEN 1 ELSE 0 END AS moe_bear_engulf,
            
            -- RAID
            CASE WHEN p.low_1 <= w.wrn_level AND p.close > w.wrn_level THEN 1 ELSE 0 END AS raid_wrn_long,
            CASE WHEN p.high_1 >= w.wrn_level AND p.close < w.wrn_level THEN 1 ELSE 0 END AS raid_wrn_short,
            
            -- SWEEP + SHIFT
            CASE WHEN p.low_1 <= p.low AND p.close > p.high_1 THEN 1 ELSE 0 END AS sweep_shift_long,
            CASE WHEN p.high_1 >= p.high AND p.close < p.low_1 THEN 1 ELSE 0 END AS sweep_shift_short,
            
            -- Свип сессий
            CASE WHEN p.low < s.asian_low AND p.close > s.asian_low THEN 1 ELSE 0 END AS sweep_asia_low,
            CASE WHEN p.high > s.asian_high AND p.close < s.asian_high THEN 1 ELSE 0 END AS sweep_asia_high,
            CASE WHEN p.low < s.london_low AND p.close > s.london_low THEN 1 ELSE 0 END AS sweep_london_low,
            CASE WHEN p.high > s.london_high AND p.close < s.london_high THEN 1 ELSE 0 END AS sweep_london_high,
            
            -- Сессия (1=Asia, 2=London, 3=NY)
            CASE 
                WHEN EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= 3 AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < 11 THEN 1
                WHEN EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= 12 AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < 16 THEN 2
                ELSE 3 
            END AS session_id
            
        FROM with_prev p
        LEFT JOIN wrn_levels w ON p.bar_time = w.bar_time
        LEFT JOIN session_ranges s ON date_trunc('day', p.bar_time + {time_offset_sql}) = s.adj_day
    )
    SELECT DISTINCT ON (signal_time) * FROM raw_signals 
    WHERE (raid_wrn_long = 1 OR raid_wrn_short = 1) 
      AND (moe_bull_engulf=1 OR moe_bear_engulf=1)
    ORDER BY signal_time;
    """
    signals = con.execute(query_dataset).fetchdf()
    
    if len(signals) < 100:
        print(f"Недостаточно данных для {SYMBOL} (найдено {len(signals)}).")
        return

    bars_df = con.execute(f"""
        SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time, 
               arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close
        FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
    """).fetchdf()
    bars_df.set_index('bar_time', inplace=True)

    X_data, Y_data, details = [], [], []
    
    for idx, row in signals.iterrows():
        entry = row['entry_price']
        signal_idx = bars_df.index.get_loc(row['signal_time'])
        if signal_idx < 1: continue
        
        prev_bar = bars_df.iloc[signal_idx - 1]
        direction = 1 if row['sweep_shift_long'] == 1 else -1
        
        if direction == 1:
            sl = prev_bar['low'] - (2 * pip_size)
            sl_dist = entry - sl
            tp = entry + sl_dist
        else:
            sl = prev_bar['high'] + (2 * pip_size)
            sl_dist = sl - entry
            tp = entry - sl_dist
            
        sl_pips = sl_dist / pip_size
        if sl_pips < 5 or sl_pips > 150: continue # Для 1H стопы могут быть шире
        
        future_bars = bars_df.iloc[signal_idx + 1: signal_idx + 50] # Смотрим 50 часов вперед
        result = None
        for t, b in future_bars.iterrows():
            if direction == 1:
                if b['low'] <= sl: result = 0; break
                if b['high'] >= tp: result = 1; break
            else:
                if b['high'] >= sl: result = 0; break
                if b['low'] <= tp: result = 1; break
                
        if result is None: continue
        
        # Признаки
        X_data.append([
            row['moe_bull_engulf'], row['moe_bear_engulf'], 
            row['sweep_shift_long'], row['sweep_shift_short'],
            row['avg_spread'] / pip_size, sl_pips, row['atr'] / pip_size,
            row['sweep_asia_low'], row['sweep_asia_high'], 
            row['sweep_london_low'], row['sweep_london_high'],
            row['session_id']
        ])
        Y_data.append(result)
        details.append({"time": row['signal_time'], "year": row['signal_time'].year, "sl_pips": sl_pips, "result": result})

    X_real = np.array(X_data)
    Y_real = np.array(Y_data)
    details_df = pd.DataFrame(details)

    if len(X_real) < 100: return

    # Walk-Forward
    test_years = [2022, 2023, 2024, 2025]
    overall_trades, overall_wins, overall_pnl_usd = 0, 0, 0
    
    print(f"\nРезультаты Walk-Forward для {SYMBOL}:")
    for test_year in test_years:
        train_mask = details_df['year'] < test_year
        test_mask = details_df['year'] == test_year
        
        if sum(train_mask) < 100 or sum(test_mask) < 20: 
            continue
            
        X_train, Y_train = X_real[train_mask], Y_real[train_mask]
        X_test = X_real[test_mask]
        details_test = details_df[test_mask].reset_index(drop=True)
        
        clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
        clf.fit(X_train, Y_train)
        
        y_prob = clf.predict_proba(X_test)[:, 1]
        selected_trades = details_test[y_prob >= 0.54] 
        
        if len(selected_trades) == 0: continue
            
        wins = (selected_trades['result'] == 1).sum()
        total_pnl_usd = 0
        
        for _, trade in selected_trades.iterrows():
            lot_size = FIXED_RISK_USD / (trade['sl_pips'] * pip_val)
            lot_size = max(round(lot_size, 2), 0.01)
            comm = lot_size * COMMISSION_PER_LOT
            
            if trade['result'] == 1:
                profit = (trade['sl_pips'] * pip_val * lot_size) - comm
            else:
                profit = -(trade['sl_pips'] * pip_val * lot_size) - comm
            total_pnl_usd += profit
            
        win_rate = (wins / len(selected_trades)) * 100
        print(f"  [{test_year}] Сделок: {len(selected_trades):3d} | Винрейт: {win_rate:.2f}% | PnL: ${total_pnl_usd:.2f}")
        
        overall_trades += len(selected_trades)
        overall_wins += wins
        overall_pnl_usd += total_pnl_usd

    if overall_trades > 0:
        print(f"  ИТОГО {SYMBOL}: Сделок: {overall_trades} | Винрейт: {(overall_wins/overall_trades)*100:.2f}% | PnL: ${overall_pnl_usd:.2f}")

# Запускаем тесты
for sym, cfg in INSTRUMENTS.items():
    run_research(sym, cfg)