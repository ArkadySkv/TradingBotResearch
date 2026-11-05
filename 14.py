import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
SYMBOL = "EURUSD" 
CONFIG = {"EURUSD": {"pip_size": 0.0001, "wrn_step": 0.0010}}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')

print(f"=== Сбор ВСЕХ признаков для {SYMBOL} ===")
time_offset_sql = "INTERVAL '1 hour'" 

query_dataset = f"""
WITH raw_ticks AS (
    SELECT timestamp, bid, ask, volume,
        SUM(bid * volume) OVER (PARTITION BY date_trunc('day', timestamp) ORDER BY timestamp) / SUM(volume) OVER (PARTITION BY date_trunc('day', timestamp) ORDER BY timestamp) AS vwap,
        SUM(POWER(bid, 2) * volume) OVER (PARTITION BY date_trunc('day', timestamp) ORDER BY timestamp) / SUM(volume) OVER (PARTITION BY date_trunc('day', timestamp) ORDER BY timestamp) AS vwap_sq
    FROM read_parquet('{base_path}')
),
base_data AS (
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
        avg(ask - bid) AS avg_spread,
        arg_max(vwap, timestamp) AS vwap_close,
        arg_max(SQRT(GREATEST(vwap_sq - POWER(vwap, 2), 0)), timestamp) AS vwap_stddev
    FROM raw_ticks GROUP BY bar_time
),
with_prev AS (
    SELECT *, 
        vwap_close + (2 * vwap_stddev) AS vwap_high, vwap_close - (2 * vwap_stddev) AS vwap_low,
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
    -- ВСЕ ВАШИ ПРИЗНАКИ (Features)
    CASE WHEN p.low_2 <= w.wrn_level AND p.close > w.wrn_level THEN 1 ELSE 0 END AS is_raid_wrn,
    CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > (p.open_1+p.close_1)/2) THEN 1 ELSE 0 END AS is_ms,
    CASE WHEN (p.close_1 > p.open_1) AND (p.close < p.open) AND (p.close < (p.open_1+p.close_1)/2) THEN 1 ELSE 0 END AS is_es,
    CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS is_engulfing,
    CASE WHEN (p.close > p.high_2) THEN 1 ELSE 0 END AS is_shift_up,
    CASE WHEN (p.close < p.low_2) THEN 1 ELSE 0 END AS is_shift_down,
    CASE WHEN p.low > p.high_3 THEN 1 ELSE 0 END AS in_bullish_fvg,
    CASE WHEN p.high < p.low_3 THEN 1 ELSE 0 END AS in_bearish_fvg,
    CASE WHEN ABS(p.close - p.vwap_high) <= (3 * {cfg['pip_size']}) OR ABS(p.close - p.vwap_low) <= (3 * {cfg['pip_size']}) THEN 1 ELSE 0 END AS is_at_vwap_band,
    CASE WHEN EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= 0 AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < 3 THEN 1 ELSE 0 END AS is_dead_zone
FROM joined_htf p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.low_2 <= w.wrn_level AND p.close > w.wrn_level -- Базовое условие: мы ищем только ЛОНГИ у WRN
ORDER BY signal_time;
"""

df = con.execute(query_dataset).fetchdf()
print(f"Собрано сетапов (только лонги у WRN): {len(df):,}")

bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

targets = []
pip_size = cfg['pip_size']

for idx, row in df.iterrows():
    entry = row['entry_price']
    sl = row['low_3'] - (2 * pip_size) 
    sl_dist = entry - sl
    if sl_dist < 3 * pip_size or sl_dist > 50 * pip_size:
        targets.append(np.nan); continue
    tp = entry + (sl_dist * 1.25)
    future_bars = bars_df.loc[row['signal_time']:].iloc[1:60]
    result = np.nan
    for t, b in future_bars.iterrows():
        if b['low'] <= sl: result = 0; break
        if b['high'] >= tp: result = 1; break
    targets.append(result)

df['target'] = targets
df = df.dropna(subset=['target'])
print(f"Валидных сделок: {len(df):,}")

# Скармливаем ВСЕ признаки модели
features = [
    'avg_spread', 'hour_of_day', 'is_dead_zone', 'is_at_vwap_band', 
    'in_bullish_fvg', 'in_bearish_fvg', 'is_ms', 'is_es', 
    'is_engulfing', 'is_shift_up', 'is_shift_down'
]
X = df[features]
y = df['target'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

clf = RandomForestClassifier(n_estimators=300, max_depth=7, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)[:, 1]

print("\n--- РЕЗУЛЬТАТЫ ML (ВСЕ ПРИЗНАКИ) ---")
print("Важность признаков (Что модель считает главным):")
for feat, imp in sorted(zip(features, clf.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.3f}")

print("\n--- Поиск порога (Сделки с высокой уверенностью) ---")
for threshold in [0.50, 0.52, 0.55, 0.58]:
    high_conf_idx = y_prob >= threshold
    if sum(high_conf_idx) > 0:
        actual_winrate = y_test[high_conf_idx].mean()
        print(f"Порог {threshold:.2f} | Сделок: {sum(high_conf_idx):4d} | Винрейт: {actual_winrate:.2%}")
