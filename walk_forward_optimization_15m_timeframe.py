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
SYMBOL = "EURUSD" 
CONFIG = {"EURUSD": {"pip_size": 0.0001, "wrn_step": 0.0010}}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
time_offset_sql = "INTERVAL '1 hour'" 
COMMISSION_PER_LOT = 6.0 
TIMEFRAME = 15

print(f"=== WALK-FORWARD OPTIMIZATION (15M) ===")
print(f"=== Символ: {SYMBOL} | Комиссия: ${COMMISSION_PER_LOT} ===")

con = duckdb.connect(':memory:')

# 1. Сбор данных (15M, WRN/DO + Sweep/Shift + MoE)
query_dataset = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time,
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
        
        -- SWEEP + SHIFT
        CASE WHEN p.low_1 <= p.low AND p.close > p.high_1 THEN 1 ELSE 0 END AS sweep_shift_long,
        
        -- Sessions (1=Asia, 2=Frankfurt, 3=London, 4=NY)
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

bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# 2. Подготовка датасета (Таргеты)
X_data, Y_data, details = [], [], []
pip_size = cfg['pip_size']

for idx, row in signals.iterrows():
    entry = row['entry_price']
    signal_idx = bars_df.index.get_loc(row['signal_time'])
    if signal_idx < 1: continue
    
    prev_bar = bars_df.iloc[signal_idx - 1]
    sl = prev_bar['low'] - (2 * pip_size) 
    sl_dist = entry - sl
    sl_pips = sl_dist / pip_size
    
    if sl_pips < 3 or sl_pips > 60: continue
    
    tp = entry + (sl_dist * 1.0) # TP = 1:1
    
    future_bars = bars_df.iloc[signal_idx + 1: signal_idx + 100]
    result = None
    for t, b in future_bars.iterrows():
        if b['low'] <= sl: 
            result = 0; break
        if b['high'] >= tp: 
            result = 1; break
            
    if result is None: continue
        
    X_data.append([
        row['moe_engulf'], row['moe_doji'], row['sweep_shift_long'],
        row['session'], row['avg_spread'] / pip_size, sl_pips
    ])
    Y_data.append(result)
    details.append({
        "time": row['signal_time'], 
        "year": row['signal_time'].year,
        "sl_pips": sl_pips, 
        "result": result
    })

X_real = np.array(X_data)
Y_real = np.array(Y_data)
details_df = pd.DataFrame(details)

print(f"Всего сетапов для теста: {len(X_real)}")

# 3. WALK-FORWARD OPTIMIZATION
# Тестируем на 2022, 2023, 2024, 2025 (обучаем на всех годах ДО теста)
test_years = [2022, 2023, 2024, 2025]

overall_trades = 0
overall_wins = 0
overall_pnl_usd = 0

print("\n--- РЕЗУЛЬТАТЫ ПО ГОДАМ (Out-of-Sample) ---")

for test_year in test_years:
    train_mask = details_df['year'] < test_year
    test_mask = details_df['year'] == test_year
    
    if sum(train_mask) < 500 or sum(test_mask) < 50:
        print(f"[{test_year}] Недостаточно данных (Train: {sum(train_mask)}, Test: {sum(test_mask)}). Пропуск.")
        continue
        
    X_train, Y_train = X_real[train_mask], Y_real[train_mask]
    X_test, Y_test = X_real[test_mask], Y_real[test_mask]
    details_test = details_df[test_mask].reset_index(drop=True)
    
    # Обучаем XGBoost на прошлых годах
    clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
    clf.fit(X_train, Y_train)
    
    # Предсказываем вероятности на тестовом годе
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Фильтруем по порогу уверенности модели
    threshold = 0.52
    selected_trades = details_test[y_prob >= threshold]
    
    if len(selected_trades) == 0:
        print(f"[{test_year}] Модель не нашла уверенных сделок.")
        continue
        
    # Считаем PnL
    wins = (selected_trades['result'] == 1).sum()
    losses = (selected_trades['result'] == 0).sum()
    total_pnl_pips = 0
    
    for _, trade in selected_trades.iterrows():
        if trade['result'] == 1:
            total_pnl_pips += trade['sl_pips'] # TP = 1:1
        else:
            total_pnl_pips -= trade['sl_pips']
            
    gross_pnl_usd = total_pnl_pips * 10.0
    commission = len(selected_trades) * COMMISSION_PER_LOT
    net_pnl_usd = gross_pnl_usd - commission
    
    win_rate = (wins / len(selected_trades)) * 100 if len(selected_trades) > 0 else 0
    
    print(f"\n[ГОД: {test_year}]")
    print(f"  Train size: {sum(train_mask)} | Test size: {sum(test_mask)}")
    print(f"  Отобрано ML сделок: {len(selected_trades)}")
    print(f"  Винрейт: {win_rate:.2f}%")
    print(f"  Чистый PnL: ${net_pnl_usd:.2f}")
    
    overall_trades += len(selected_trades)
    overall_wins += wins
    overall_pnl_usd += net_pnl_usd

# Итог
print("\n" + "="*40)
print("--- ОБЩИЙ ИТОГ ЗА ВСЕ ТЕСТОВЫЕ ГОДЫ ---")
if overall_trades > 0:
    print(f"Всего сделок: {overall_trades}")
    print(f"Общий винрейт: {(overall_wins / overall_trades) * 100:.2f}%")
    print(f"СУММАРНЫЙ ЧИСТЫЙ PnL: ${overall_pnl_usd:.2f}")
else:
    print("Не было совершено ни одной сделки.")