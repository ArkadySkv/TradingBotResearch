import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
SYMBOL = "EURUSD" 
CONFIG = {"EURUSD": {"wrn_step": 0.0010, "pip_size": 0.0001}}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')

print(f"=== Сбор данных для ML v2 (Символ: {SYMBOL}) ===")
time_offset_sql = "INTERVAL '1 hour'" 

query_dataset = f"""
-- 1. Считаем дельту (Order Flow) для КАЖДОГО тика
WITH raw_ticks AS (
    SELECT 
        timestamp, bid, ask, volume,
        CASE 
            WHEN bid > LAG(bid) OVER (ORDER BY timestamp) THEN volume 
            WHEN bid < LAG(bid) OVER (ORDER BY timestamp) THEN -volume 
            ELSE 0 
        END AS tick_delta
    FROM read_parquet('{base_path}')
),
-- 2. Строим 5-минутные свечи и суммируем дельты
base_data AS (
    SELECT 
        time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, 
        max(bid) AS high, 
        min(bid) AS low, 
        arg_max(bid, timestamp) AS close, 
        avg(ask - bid) AS avg_spread,
        SUM(tick_delta) AS bar_delta
    FROM raw_ticks 
    GROUP BY bar_time
),
with_prev AS (
    SELECT *, 
        LAG(open) OVER (ORDER BY bar_time) AS open_1, LAG(close) OVER (ORDER BY bar_time) AS close_1,
        LAG(open, 2) OVER (ORDER BY bar_time) AS open_2, LAG(close, 2) OVER (ORDER BY bar_time) AS close_2,
        LAG(low, 2) OVER (ORDER BY bar_time) AS low_2, LAG(high, 2) OVER (ORDER BY bar_time) AS high_2,
        LAG(low, 3) OVER (ORDER BY bar_time) AS low_3, LAG(high, 3) OVER (ORDER BY bar_time) AS high_3
    FROM base_data
),
htf_4h AS (
    SELECT time_bucket(INTERVAL '4 hours', timestamp) AS htf_time,
        max(bid) AS high_4h, min(bid) AS low_4h
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
    p.bar_delta,
    EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) AS hour_of_day,
    p.low_2, p.high_2, p.low_3, p.high_3,
    CASE WHEN p.low > p.high_3 THEN 1 ELSE 0 END AS in_bullish_fvg_4h,
    CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS is_engulfing,
    CASE WHEN (p.close > p.high_2) THEN 1 ELSE 0 END AS is_shift
FROM joined_htf p
WHERE p.close > p.high_2 
ORDER BY signal_time;
"""

df = con.execute(query_dataset).fetchdf()
print(f"Собрано сетапов: {len(df):,}")

# Загружаем 5M свечи для симуляции
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
print(f"Валидных сделок для ML: {len(df):,}")

# Обучаем модель
features = ['avg_spread', 'bar_delta', 'hour_of_day', 'in_bullish_fvg_4h', 'is_engulfing', 'is_shift']
X = df[features]
y = df['target'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

clf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)[:, 1]

print("\n--- РЕЗУЛЬТАТЫ ML v2 ---")
print("Важность признаков:")
for feat, imp in sorted(zip(features, clf.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.3f}")

# Проверяем винрейт при разных порогах уверенности
print("\n--- Поиск оптимального порога ---")
for threshold in [0.50, 0.52, 0.55, 0.58, 0.60]:
    high_conf_idx = y_prob >= threshold
    if sum(high_conf_idx) > 0:
        actual_winrate = y_test[high_conf_idx].mean()
        print(f"Порог {threshold:.2f} | Сделок: {sum(high_conf_idx):4d} | Винрейт: {actual_winrate:.2%}")
