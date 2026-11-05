import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
SYMBOL = "EURUSD" 
CONFIG = {"EURUSD": {"pip_size": 0.0001, "wrn_step": 0.0010}}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
time_offset_sql = "INTERVAL '1 hour'" 
COMMISSION_PER_LOT = 6.0 

def run_backtest(timeframe_minutes):
    print(f"\n{'='*60}")
    print(f"=== ML-ФИЛЬТР ДЛЯ {timeframe_minutes}M ТАЙМФРЕЙМА ===")
    print(f"{'='*60}")
    
    con = duckdb.connect(':memory:')
    
    # SQL: Собираем все базовые сетапы (RAID + MoE + Sweep)
    query_dataset = f"""
    WITH base_data AS (
        SELECT time_bucket(INTERVAL '{timeframe_minutes} minutes', timestamp) AS bar_time,
            arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
            avg(ask - bid) AS avg_spread
        FROM read_parquet('{base_path}') GROUP BY bar_time
    ),
    weekly_range AS (
        SELECT date_trunc('week', bar_time) AS week_start, max(high) AS prev_high, min(low) AS prev_low FROM base_data GROUP BY week_start
    ),
    wrn_levels AS (
        SELECT b.bar_time, CAST(t.level_int AS DOUBLE) * {cfg['wrn_step']} AS wrn_level
        FROM base_data b
        JOIN weekly_range w ON date_trunc('week', b.bar_time) = w.week_start + INTERVAL '7 days'
        CROSS JOIN UNNEST(generate_series(
            CAST(floor(w.prev_low / {cfg['wrn_step']}) AS BIGINT) - 1, 
            CAST(ceil(w.prev_high / {cfg['wrn_step']}) AS BIGINT) + 1, 
            1
        )) AS t(level_int)
    ),
    session_opens AS (
        SELECT date_trunc('day', bar_time + {time_offset_sql}) AS adj_day, arg_min(open, bar_time) AS do_level
        FROM base_data WHERE EXTRACT(hour FROM (bar_time + {time_offset_sql})) = 16 GROUP BY adj_day
    ),
    with_prev AS (
        SELECT *, 
            LAG(open) OVER (ORDER BY bar_time) AS open_1, 
            LAG(close) OVER (ORDER BY bar_time) AS close_1,
            LAG(high) OVER (ORDER BY bar_time) AS high_1,
            LAG(low) OVER (ORDER BY bar_time) AS low_1
        FROM base_data
    ),
    raw_signals AS (
        SELECT 
            p.bar_time AS signal_time, p.close AS entry_price, p.avg_spread, p.low, p.high,
            
            -- MoE
            CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS moe_engulf,
            CASE WHEN ABS(p.close - p.open) <= (p.high - p.low) * 0.25 AND (p.high - p.low) > 0 THEN 1 ELSE 0 END AS moe_doji,
            
            -- RAID
            CASE WHEN p.low_1 <= w.wrn_level AND p.close > w.wrn_level THEN 1 ELSE 0 END AS raid_wrn,
            CASE WHEN p.low_1 <= s.do_level AND p.close > s.do_level THEN 1 ELSE 0 END AS raid_do,
            
            -- SWEEP (касание тени предыдущей)
            CASE WHEN p.low_1 <= p.low AND p.close > p.high_1 THEN 1 ELSE 0 END AS sweep_shift_long,
            
            -- Sessions
            CASE 
                WHEN EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= 3 AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < 11 THEN 1
                WHEN EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= 11 AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < 12 THEN 2
                WHEN EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= 12 AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < 16 THEN 3
                ELSE 4 
            END AS session
        FROM with_prev p
        LEFT JOIN wrn_levels w ON p.bar_time = w.bar_time
        LEFT JOIN session_opens s ON date_trunc('day', p.bar_time + {time_offset_sql}) = s.adj_day
    )
    SELECT DISTINCT ON (signal_time) * FROM raw_signals 
    WHERE (raid_wrn = 1 OR raid_do = 1) 
      AND (moe_engulf=1 OR moe_doji=1)
    ORDER BY signal_time;
    """
    signals = con.execute(query_dataset).fetchdf()
    
    if len(signals) == 0: return

    bars_df = con.execute(f"""
        SELECT time_bucket(INTERVAL '{timeframe_minutes} minutes', timestamp) AS bar_time, 
               arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close
        FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
    """).fetchdf()
    bars_df.set_index('bar_time', inplace=True)

    # Формируем датасет с динамическим SL и TP 1:1 (для честной оценки)
    X_data, Y_data, details = [], [], []
    pip_size = cfg['pip_size']

    for idx, row in signals.iterrows():
        entry = row['entry_price']
        signal_idx = bars_df.index.get_loc(row['signal_time'])
        if signal_idx < 1: continue
        
        prev_bar = bars_df.iloc[signal_idx - 1]
        # Динамический стоп за минимум свипа
        sl = prev_bar['low'] - (2 * pip_size) 
        sl_dist = entry - sl
        sl_pips = sl_dist / pip_size
        
        if sl_pips < 3 or sl_pips > 60: continue # Отсекаем совсем мусорные стопы
        
        # TP = 1:1 (самый честный вариант для начала)
        tp = entry + (sl_dist * 1.0) 
        
        future_bars = bars_df.iloc[signal_idx + 1: signal_idx + 100]
        result = None
        for t, b in future_bars.iterrows():
            if b['low'] <= sl: 
                result = 0
                break
            if b['high'] >= tp: 
                result = 1
                break
                
        if result is None: continue
            
        # Признаки для ML (X)
        X_data.append([
            row['moe_engulf'], row['moe_doji'], row['sweep_shift_long'],
            row['session'], row['avg_spread'] / pip_size, sl_pips
        ])
        Y_data.append(result)
        details.append({"time": row['signal_time'], "entry": entry, "sl_pips": sl_pips, "result": result})

    if len(X_data) < 100: return

    X_real = np.array(X_data)
    Y_real = np.array(Y_data)
    
    # Обучаем XGBoost на 70% данных, тестируем на 30% (Out-of-Sample)
    split_idx = int(len(X_real) * 0.7)
    X_train, X_test = X_real[:split_idx], X_real[split_idx:]
    Y_train, Y_test = Y_real[:split_idx], Y_real[split_idx:]
    details_test = details[split_idx:]

    clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
    clf.fit(X_train, Y_train)

    # Предсказываем вероятности на тесте
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Фильтруем сделки (берем только те, где модель уверена > 0.52)
    threshold = 0.52
    selected_trades = [details_test[i] for i in range(len(details_test)) if y_prob[i] >= threshold]

    if not selected_trades:
        print(f"[{timeframe_minutes}M] Модель не нашла уверенных сделок.")
        return

    # Считаем PnL
    wins, losses = 0, 0
    total_pnl_pips = 0
    
    for trade in selected_trades:
        if trade['result'] == 1:
            wins += 1
            total_pnl_pips += trade['sl_pips'] # TP = 1:1
        else:
            losses += 1
            total_pnl_pips -= trade['sl_pips']

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100
    gross_pnl_usd = total_pnl_pips * 10.0
    commission = total_trades * COMMISSION_PER_LOT
    net_pnl_usd = gross_pnl_usd - commission

    print(f"Всего сетапов: {len(X_real)} | Отфильтровано ML: {total_trades}")
    print(f"Винрейт: {win_rate:.2f}%")
    print(f"Чистый PnL (с комиссией): ${net_pnl_usd:.2f}")
    
# Запускаем для всех таймфреймов
for tf in [1, 3, 5, 15, 30, 45, 60, 120, 240, 360]:
    run_backtest(tf)