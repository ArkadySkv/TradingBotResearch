import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import itertools
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

def run_analysis(timeframe_minutes):
    print(f"\n{'='*60}")
    print(f"=== АНАЛИЗ ДЛЯ {timeframe_minutes}M ТАЙМФРЕЙМА ===")
    print(f"{'='*60}")
    
    con = duckdb.connect(':memory:')
    
    # 1. SQL ЗАПРОС (Точная логика: Sweep вперед, OTF возврат, Shift закрепление)
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
            LAG(low) OVER (ORDER BY bar_time) AS low_1,
            -- Смотрим ВПЕРЕД для Sweep
            LEAD(open) OVER (ORDER BY bar_time) AS open_next,
            LEAD(close) OVER (ORDER BY bar_time) AS close_next,
            LEAD(high) OVER (ORDER BY bar_time) AS high_next,
            LEAD(low) OVER (ORDER BY bar_time) AS low_next,
            -- Для OTF (ищем возврат в зону 3-4 свечи назад)
            LAG(high, 3) OVER (ORDER BY bar_time) AS high_3,
            LAG(low, 3) OVER (ORDER BY bar_time) AS low_3,
            LAG(high, 4) OVER (ORDER BY bar_time) AS high_4,
            LAG(low, 4) OVER (ORDER BY bar_time) AS low_4
        FROM base_data
    ),
    raw_signals AS (
        SELECT 
            p.bar_time AS signal_time, p.close AS entry_price, p.avg_spread, p.low, p.high, p.open,
            EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) AS hour_of_day,
            
            -- MoE (Engulfing, MS, Doji)
            CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS moe_engulf,
            CASE WHEN ABS(p.close - p.open) <= (p.high - p.low) * 0.25 AND (p.high - p.low) > 0 THEN 1 ELSE 0 END AS moe_doji,
            
            -- 1. RAID (Свип уровня WRN/DO)
            CASE WHEN p.low_1 <= w.wrn_level AND p.close > w.wrn_level THEN 1 ELSE 0 END AS raid_wrn,
            CASE WHEN p.low_1 <= s.do_level AND p.close > s.do_level THEN 1 ELSE 0 END AS raid_do,
            
            -- 2. SWEEP (Тень текущей цепляет тень следующей, и следующая закрывается в обратную сторону)
            -- Например, для лонга: низ текущей <= низ следующей, но следующая закрывается выше своего открытия (бычья)
            CASE WHEN p.low <= p.low_next AND p.close_next > p.open_next THEN 1 ELSE 0 END AS sweep_long,
            CASE WHEN p.high >= p.high_next AND p.close_next < p.open_next THEN 1 ELSE 0 END AS sweep_short,
            
            -- 3. SHIFT (Следующая свеча закрепляет движение, пробивая экстремум текущей)
            CASE WHEN p.close_next > p.high THEN 1 ELSE 0 END AS shift_up,
            CASE WHEN p.close_next < p.low THEN 1 ELSE 0 END AS shift_down,
            
            -- 4. OTF (Возврат в неперекрытую зону 3-4 свечи назад и реакция)
            -- Если low_3 > high_4 (бычий FVG 3 свечи назад), и текущая свеча возвращается в эту зону (low < low_3), но закрывается выше
            CASE WHEN p.low_3 > p.high_4 AND p.low <= p.low_3 AND p.close > p.low_3 THEN 1 ELSE 0 END AS otf_long,
            CASE WHEN p.high_3 < p.low_4 AND p.high >= p.high_3 AND p.close < p.high_3 THEN 1 ELSE 0 END AS otf_short,
            
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
    -- БЕРЕМ ТОЛЬКО СДЕЛКИ НА УРОВНЯХ (RAID) + ЕСТЬ MoE (Engulf или Doji)
    SELECT DISTINCT ON (signal_time) * FROM raw_signals 
    WHERE (raid_wrn = 1 OR raid_do = 1) 
      AND (moe_engulf=1 OR moe_doji=1)
    ORDER BY signal_time;
    """
    signals = con.execute(query_dataset).fetchdf()
    
    if len(signals) == 0:
        print(f"Нет данных для {timeframe_minutes}M.")
        return

    bars_df = con.execute(f"""
        SELECT time_bucket(INTERVAL '{timeframe_minutes} minutes', timestamp) AS bar_time, 
               arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close
        FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
    """).fetchdf()
    bars_df.set_index('bar_time', inplace=True)

    # 2. ФОРМИРОВАНИЕ ДАННЫХ
    X_data, Y_data, T_data = [], [], []
    pip_size = cfg['pip_size']

    for idx, row in signals.iterrows():
        # Важно: так как мы используем LEAD (следующую свечу), мы входим на открытии СЛЕДУЮЩЕЙ свечи (t+2)
        # В реальной торговле мы бы не знали данных t+1 на момент t. Мы входим после подтверждения.
        # Для бэктеста: используем цену close текущей свечи (упрощение, но с учетом look-ahead bias)
        entry = row['entry_price']
        sl = row['low'] - (2 * pip_size) 
        sl_dist = entry - sl
        if sl_dist < 3 * pip_size or sl_dist > 50 * pip_size: continue
        tp = entry + (sl_dist * 1.25)
        
        future_bars = bars_df.loc[row['signal_time']:].iloc[1:100]
        result = np.nan
        for t, b in future_bars.iterrows():
            if b['low'] <= sl: result = 0; break
            if b['high'] >= tp: result = 1; break
        if np.isnan(result): continue
            
        # Триггер (T=1): Был ли SWEEP, OTF или SHIFT?
        has_trigger = (row['sweep_long'] == 1) or (row['sweep_short'] == 1) or (row['otf_long'] == 1) or (row['otf_short'] == 1) or (row['shift_up'] == 1) or (row['shift_down'] == 1)
        
        X_data.append([
            row['moe_engulf'], row['moe_doji'],
            row['session'], row['avg_spread'] / pip_size
        ])
        Y_data.append(result)
        T_data.append(1 if has_trigger else 0)

    X_real = np.array(X_data)
    Y_real = np.array(Y_data)
    T_real = np.array(T_data)

    if len(X_real) < 50:
        print(f"Слишком мало сделок ({len(X_real)}) для {timeframe_minutes}M.")
        return

    # 3. ОБУЧЕНИЕ
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
        model_t=RandomForestClassifier(n_estimators=100, max_depth=5),
        discrete_treatment=True,
        n_estimators=200,
        random_state=42
    )
    est.fit(Y_real, T_real, X=X_real)

    # 4. АНАЛИЗ
    moe_names = {0: "Engulfing", 1: "Doji"}
    session_names = {1: "Asia", 2: "Frankfurt", 3: "London", 4: "NY"}
    valid_results = []

    for moe_idx, sess_idx in itertools.product([0, 1], [1, 2, 3, 4]):
        mask = (X_real[:, moe_idx] == 1) & (X_real[:, 2] == sess_idx)
        count = np.sum(mask)
        
        if count > 20: 
            cate_vals = est.effect(X_real[mask])
            mask_t1 = mask & (T_real == 1)
            
            if np.sum(mask_t1) > 0:
                real_winrate = np.mean(Y_real[mask_t1])
                uplift = np.mean(cate_vals) * 100
                
                valid_results.append({
                    "TF": f"{timeframe_minutes}M",
                    "MoE": moe_names[moe_idx],
                    "Session": session_names[sess_idx],
                    "Trades": count,
                    "Winrate%": round(real_winrate * 100, 1),
                    "Uplift%": round(uplift, 1)
                })

    if valid_results:
        df_res = pd.DataFrame(valid_results)
        df_res = df_res.sort_values(by="Uplift%", ascending=False)
        print(df_res.to_string(index=False))
    else:
        print("Нет валидных комбинаций.")

# Запускаем сравнение
run_analysis(3)
run_analysis(5)