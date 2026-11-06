import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
SYMBOL = "EURUSD" 
CONFIG = {"EURUSD": {"pip_size": 0.0001, "wrn_step": 0.0010}}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')

print(f"=== ML v3 (15M TF + XGBoost) для {SYMBOL} ===")
time_offset_sql = "INTERVAL '1 hour'" 

query_dataset = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '15 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
        avg(ask - bid) AS avg_spread
    FROM read_parquet('{base_path}') GROUP BY bar_time
),
with_prev AS (
    SELECT *, 
        LAG(open) OVER (ORDER BY bar_time) AS open_1, LAG(close) OVER (ORDER BY bar_time) AS close_1,
        LAG(open, 2) OVER (ORDER BY bar_time) AS open_2, LAG(close, 2) OVER (ORDER BY bar_time) AS close_2,
        LAG(high, 2) OVER (ORDER BY bar_time) AS high_2, LAG(low, 2) OVER (ORDER BY bar_time) AS low_2,
        LAG(high, 3) OVER (ORDER BY bar_time) AS high_3, LAG(low, 3) OVER (ORDER BY bar_time) AS low_3
    FROM base_data
),
weekly_range AS (
    SELECT date_trunc('week', bar_time) AS week_start, max(high) AS prev_high, min(low) AS prev_low FROM with_prev GROUP BY week_start
),
wrn_levels AS (
    SELECT b.bar_time, CAST(t.level_int AS DOUBLE) * {cfg['wrn_step']} AS wrn_level
    FROM with_prev b
    JOIN weekly_range w ON date_trunc('week', b.bar_time) = w.week_start + INTERVAL '7 days'
    CROSS JOIN UNNEST(generate_series(CAST(floor(w.prev_low / {cfg['wrn_step']}) AS BIGINT), CAST(ceil(w.prev_high / {cfg['wrn_step']}) AS BIGINT), 1)) AS t(level_int)
),
htf_4h AS (
    SELECT time_bucket(INTERVAL '4 hours', timestamp) AS htf_time, max(bid) AS high_4h, min(bid) AS low_4h
    FROM read_parquet('{base_path}') GROUP BY htf_time
),
joined_htf AS (
    SELECT l.*, h.high_4h, h.low_4h
    FROM with_prev l
    LEFT JOIN htf_4h h ON l.bar_time >= h.htf_time AND l.bar_time < h.htf_time + INTERVAL '4 hours'
)
SELECT 
    p.bar_time AS signal_time,
    p.close AS entry_price,
    p.avg_spread,
    EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) AS hour_of_day,
    p.low_3, p.high_3,
    -- Признаки
    CASE WHEN p.low_2 <= w.wrn_level AND p.close > w.wrn_level THEN 1 ELSE 0 END AS is_raid_wrn,
    CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > (p.open_1+p.close_1)/2) THEN 1 ELSE 0 END AS is_ms,
    CASE WHEN (p.close_1 > p.open_1) AND (p.close < p.open) AND (p.close < (p.open_1+p.close_1)/2) THEN 1 ELSE 0 END AS is_es,
    CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS is_engulfing,
    CASE WHEN (p.close > p.high_2) THEN 1 ELSE 0 END AS is_shift_up,
    CASE WHEN p.low > p.high_3 THEN 1 ELSE 0 END AS in_bullish_fvg,
    CASE WHEN EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= 0 AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < 3 THEN 1 ELSE 0 END AS is_dead_zone
FROM joined_htf p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.low_2 <= w.wrn_level AND p.close > w.wrn_level
ORDER BY signal_time;
"""

df = con.execute(query_dataset).fetchdf()
print(f"Собрано сетапов (15M): {len(df):,}")

# Свечи для симуляции (15M)
bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '15 minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

targets = []
pip_size = cfg['pip_size']
MAX_BARS_HOLD = 16 # 16 * 15M = 4 часа

for idx, row in df.iterrows():
    entry = row['entry_price']
    sl = row['low_3'] - (2 * pip_size) 
    sl_dist = entry - sl
    if sl_dist < 5 * pip_size or sl_dist > 60 * pip_size:
        targets.append(np.nan); continue
    
    tp = entry + (sl_dist * 1.25)
    future_bars = bars_df.loc[row['signal_time']:].iloc[1:MAX_BARS_HOLD]
    
    result = np.nan
    for t, b in future_bars.iterrows():
        if b['low'] <= sl: 
            result = 0; break # LOSS
        if b['high'] >= tp: 
            result = 1; break # WIN
            
    # Если за 4 часа не достигли ни TP, ни SL - считаем это убытком (время вышло)
    if np.isnan(result) and len(future_bars) == MAX_BARS_HOLD:
        if future_bars.iloc[-1]['close'] > entry:
            result = 1 # Закрылись в плюс (хоть и меньше TP)
        else:
            result = 0 # Закрылись в минус
    targets.append(result)

df['target'] = targets
df = df.dropna(subset=['target'])
print(f"Валидных сделок: {len(df):,}")

# Обучаем XGBoost
features = [
    'avg_spread', 'hour_of_day', 'is_dead_zone', 'in_bullish_fvg', 
    'is_ms', 'is_es', 'is_engulfing', 'is_shift_up', 'is_raid_wrn'
]
X = df[features]
y = df['target'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

# scale_pos_weight помогает XGBoost, если убытков больше, чем прибылей
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1

clf = xgb.XGBClassifier(
    n_estimators=200, 
    max_depth=4, 
    learning_rate=0.05, 
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)[:, 1]

print("\n--- РЕЗУЛЬТАТЫ XGBoost (15M) ---")
print("Важность признаков:")
for feat, imp in sorted(zip(features, clf.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.3f}")

print("\n--- Поиск порога ---")
for threshold in [0.45, 0.50, 0.55, 0.60]:
    high_conf_idx = y_prob >= threshold
    if sum(high_conf_idx) > 0:
        actual_winrate = y_test[high_conf_idx].mean()
        print(f"Порог {threshold:.2f} | Сделок: {sum(high_conf_idx):4d} | Винрейт: {actual_winrate:.2%}")
