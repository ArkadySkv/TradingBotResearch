import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

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

# Мы убрали жесткий фильтр сессии NY, чтобы модель сама увидела, где лучше.
query_dataset = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
        avg(ask - bid) AS avg_spread,
        -- Прокси для Order Flow (если цена идет вверх, покупатели активны)
        SUM(CASE WHEN bid > LAG(bid) OVER (ORDER BY timestamp) THEN volume ELSE -volume END) AS bar_delta
    FROM read_parquet('{base_path}') GROUP BY bar_time
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
    -- FVG (цена в зоне неэффективности)
    CASE WHEN p.low > p.high_3 THEN 1 ELSE 0 END AS in_bullish_fvg_4h,
    -- Паттерны
    CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS is_engulfing,
    CASE WHEN (p.close > p.high_2) THEN 1 ELSE 0 END AS is_shift
FROM joined_htf p
WHERE p.close > p.high_2 -- Базовое условие, что мы вообще ищем лонг
ORDER BY signal_time;
"""

df = con.execute(query_dataset).fetchdf()
print(f"Собрано сетапов: {len(df):,}")

bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# Создаем Target (1 = WIN, 0 = LOSS)
targets = []
pip_size = cfg['pip_size']

for idx, row in df.iterrows():
    entry = row['entry_price']
    sl = row['low_3'] - (2 * pip_size) # Стоп за 3-мя свечами
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

# Обучаем модель на всех признаках
features = ['avg_spread', 'bar_delta', 'hour_of_day', 'in_bullish_fvg_4h', 'is_engulfing', 'is_shift']
X = df[features]
y = df['target'].astype(int)

# Важно: shuffle=False, чтобы тест был на будущем (Out-of-Sample)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

clf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Предсказываем вероятности (а не жесткие 0/1)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\n--- РЕЗУЛЬТАТЫ ML v2 ---")
print("Важность признаков:")
for feat, imp in sorted(zip(features, clf.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.3f}")

# Смотрим, что будет, если брать только сделки с высокой уверенностью модели (>0.55)
threshold = 0.55
high_conf_idx = y_prob >= threshold
if sum(high_conf_idx) > 0:
    actual_winrate = y_test[high_conf_idx].mean()
    print(f"\nЕсли брать только сделки с уверенностью >{threshold}:")
    print(f"  Найдено сделок: {sum(high_conf_idx)}")
    print(f"  Фактический винрейт: {actual_winrate:.2%}")
else:
    print("\nМодель не нашла ни одной сделки с высокой уверенностью.")

# Проверим базовый винрейт на тесте (без ML)
print(f"\nБазовый винрейт на всем тесте (без ML): {y_test.mean():.2%}")
