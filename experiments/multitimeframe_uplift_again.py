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
    print(f"=== АНАЛИЗ ДЛЯ {timeframe_minutes}M (Sweep + Shift, TP=1:1) ===")
    print(f"{'='*60}")
    
    con = duckdb.connect(':memory:')
    
    # SQL: Свип + Слом структуры (Shift) + Вход по закрытию
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
            
            -- SWEEP + SHIFT (Тень пробила низ, но текущая закрылась ВЫШЕ максимума предыдущей)
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
    
    if len(signals) == 0:
        return

    bars_df = con.execute(f"""
        SELECT time_bucket(INTERVAL '{timeframe_minutes} minutes', timestamp) AS bar_time, 
               arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close
        FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
    """).fetchdf()
    bars_df.set_index('bar_time', inplace=True)

    # 2. ФОРМИРОВАНИЕ ДАННЫХ (TP = 1:1)
    X_data, Y_data, T_data = [], [], []
    pip_size = cfg['pip_size']

    for idx, row in signals.iterrows():
        entry = row['entry_price']
        signal_idx = bars_df.index.get_loc(row['signal_time'])
        if signal_idx < 1: continue
        
        prev_bar = bars_df.iloc[signal_idx - 1]
        sl = prev_bar['low'] - (2 * pip_size) 
        sl_dist = entry - sl
        
        if sl_dist < 3 * pip_size or sl_dist > 60 * pip_size: continue # Для 1H стопы могут быть шире
        
        # ИЗМЕНЕНО: TP = 1:1 (вместо 1.25)
        tp = entry + (sl_dist * 1.0) 
        
        future_bars = bars_df.iloc[signal_idx + 1: signal_idx + 100]
        result = np.nan
        for t, b in future_bars.iterrows():
            if b['low'] <= sl: result = 0; break
            if b['high'] >= tp: result = 1; break
        if np.isnan(result): continue
            
        X_data.append([
            row['moe_engulf'], row['moe_doji'],
            row['session'], row['avg_spread'] / pip_size
        ])
        Y_data.append(result)
        T_data.append(row['sweep_shift_long'])

    X_real = np.array(X_data)
    Y_real = np.array(Y_data)
    T_real = np.array(T_data)

    if len(X_real) < 30 or np.sum(T_real == 1) < 30:
        print(f"Мало сделок ({len(X_real)}) для {timeframe_minutes}M.")
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
        
        if count > 15: 
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

# Запускаем только для старших таймфреймов, где был найден Edge
run_analysis(15)
run_analysis(30)
run_analysis(60)