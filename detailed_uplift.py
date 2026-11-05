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
    
    # 1. СЛОЖНЫЙ SQL: 4H FVG (жизнь 48ч) + OTF (жизнь 2ч) + Sweep
    query_dataset = f"""
    WITH raw_data AS (
        SELECT timestamp, bid, ask, volume FROM read_parquet('{base_path}')
    ),
    -- 1. Строим 4H свечи для поиска FVG
    htf_4h AS (
        SELECT time_bucket(INTERVAL '4 hours', timestamp) AS htf_time,
            arg_min(bid, timestamp) AS open_4h, max(bid) AS high_4h, min(bid) AS low_4h, arg_max(bid, timestamp) AS close_4h
        FROM raw_data GROUP BY htf_time
    ),
    -- Ищем 4H FVG (разрыв между 1-й и 3-й свечой)
    fvg_4h_raw AS (
        SELECT htf_time AS fvg_time, 
            LAG(high_4h, 2) OVER (ORDER BY htf_time) AS prev_high_2,
            LAG(low_4h, 2) OVER (ORDER BY htf_time) AS prev_low_2,
            low_4h AS current_low,
            high_4h AS current_high
        FROM htf_4h
    ),
    fvg_4h AS (
        SELECT 
            fvg_time,
            prev_high_2 AS fvg_top_bull, -- Бычий FVG: low 3-й свечи > high 1-й свечи
            current_low AS fvg_bottom_bull,
            prev_low_2 AS fvg_bottom_bear, -- Медвежий FVG: high 3-й свечи < low 1-й свечи
            current_high AS fvg_top_bear
        FROM fvg_4h_raw
        WHERE current_low > prev_high_2 OR current_high < prev_low_2
    ),
    -- 2. Строим базовые данные (3M/5M)
    base_data AS (
        SELECT time_bucket(INTERVAL '{timeframe_minutes} minutes', timestamp) AS bar_time,
            arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
            avg(ask - bid) AS avg_spread
        FROM raw_data GROUP BY bar_time
    ),
    -- Уровни
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
            LAG(open) OVER (ORDER BY bar_time) AS open_1, LAG(close) OVER (ORDER BY bar_time) AS close_1,
            LAG(high) OVER (ORDER BY bar_time) AS high_1, LAG(low) OVER (ORDER BY bar_time) AS low_1,
            LEAD(open) OVER (ORDER BY bar_time) AS open_next, LEAD(close) OVER (ORDER BY bar_time) AS close_next,
            LEAD(high) OVER (ORDER BY bar_time) AS high_next, LEAD(low) OVER (ORDER BY bar_time) AS low_next,
            -- Окно для OTF (возврат в зону 5M за последние 24 свечи = 2 часа)
            LAG(high, 12) OVER (ORDER BY bar_time) AS high_12, LAG(low, 12) OVER (ORDER BY bar_time) AS low_12,
            LAG(high, 24) OVER (ORDER BY bar_time) AS high_24, LAG(low, 24) OVER (ORDER BY bar_time) AS low_24
        FROM base_data
    ),
    raw_signals AS (
        SELECT 
            p.bar_time AS signal_time, p.close AS entry_price, p.avg_spread, p.low, p.high,
            EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) AS hour_of_day,
            
            -- MoE
            CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS moe_engulf,
            CASE WHEN ABS(p.close - p.open) <= (p.high - p.low) * 0.25 AND (p.high - p.low) > 0 THEN 1 ELSE 0 END AS moe_doji,
            
            -- RAID (Обязательное условие)
            CASE WHEN p.low_1 <= w.wrn_level AND p.close > w.wrn_level THEN 1 ELSE 0 END AS raid_wrn,
            CASE WHEN p.low_1 <= s.do_level AND p.close > s.do_level THEN 1 ELSE 0 END AS raid_do,
            
            -- ТРИГГЕР 1: Sweep (Тень касается следующей)
            CASE WHEN p.low <= p.low_next AND p.close_next > p.open_next THEN 1 ELSE 0 END AS sweep_long,
            
            -- ТРИГГЕР 2: OTF (Возврат в неперекрытую зону за последние 2 часа)
            CASE WHEN p.low_12 > p.high_24 AND p.low <= p.low_12 AND p.close > p.low_12 THEN 1 ELSE 0 END AS otf_long,
            
            -- ТРИГГЕР 3: 4H FVG (Цена зашла в зону FVG, которая была не более 48ч назад)
            -- Для простоты используем ASOF JOIN в отдельном CTE (см. ниже)
            
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
    ),
    -- Привязка FVG к сигналам (поиск активного FVG на момент сделки)
    signals_with_fvg AS (
        SELECT 
            r.*,
            -- Если цена находится внутри бычьего FVG и FVG был создан не более 48ч назад
            CASE WHEN EXISTS (
                SELECT 1 FROM fvg_4h f 
                WHERE f.fvg_time <= r.signal_time 
                  AND f.fvg_time >= r.signal_time - INTERVAL '48 hours'
                  AND r.low <= f.fvg_top_bull AND r.high >= f.fvg_bottom_bull
            ) THEN 1 ELSE 0 END AS in_4h_fvg
        FROM raw_signals r
    )
    SELECT DISTINCT ON (signal_time) * FROM signals_with_fvg 
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

    # 2. ФОРМИРОВАНИЕ ДАННЫХ (Отдельные Treatment для каждого триггера)
    X_data, Y_data, T_sweep, T_otf, T_fvg = [], [], [], [], []
    pip_size = cfg['pip_size']

    for idx, row in signals.iterrows():
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
            
        X_data.append([
            row['moe_engulf'], row['moe_doji'],
            row['session'], row['avg_spread'] / pip_size
        ])
        Y_data.append(result)
        T_sweep.append(row['sweep_long'])
        T_otf.append(row['otf_long'])
        T_fvg.append(row['in_4h_fvg'])

    X_real = np.array(X_data)
    Y_real = np.array(Y_data)
    T_sweep = np.array(T_sweep)
    T_otf = np.array(T_otf)
    T_fvg = np.array(T_fvg)

    if len(X_real) < 50:
        print(f"Слишком мало сделок ({len(X_real)}) для {timeframe_minutes}M.")
        return

    # 3. ОБУЧЕНИЕ И СРАВНЕНИЕ ТРИГГЕРОВ
    moe_names = {0: "Engulfing", 1: "Doji"}
    session_names = {1: "Asia", 2: "Frankfurt", 3: "London", 4: "NY"}
    valid_results = []

    triggers = {
        "Sweep": T_sweep,
        "OTF (2h)": T_otf,
        "4H FVG (48h)": T_fvg
    }

    for trig_name, T_data in triggers.items():
        if np.sum(T_data == 1) < 50: continue # Пропускаем, если триггеров слишком мало
        
        print(f"Обучение для триггера: {trig_name}...")
        est = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
            model_t=RandomForestClassifier(n_estimators=100, max_depth=5),
            discrete_treatment=True,
            n_estimators=200,
            random_state=42
        )
        est.fit(Y_real, T_data, X=X_real)

        for moe_idx, sess_idx in itertools.product([0, 1], [1, 2, 3, 4]):
            mask = (X_real[:, moe_idx] == 1) & (X_real[:, 2] == sess_idx)
            count = np.sum(mask)
            
            if count > 20: 
                cate_vals = est.effect(X_real[mask])
                mask_t1 = mask & (T_data == 1)
                
                if np.sum(mask_t1) > 0:
                    real_winrate = np.mean(Y_real[mask_t1])
                    uplift = np.mean(cate_vals) * 100
                    
                    valid_results.append({
                        "TF": f"{timeframe_minutes}M",
                        "Trigger": trig_name,
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