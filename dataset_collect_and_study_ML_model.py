import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
SYMBOL = "EURUSD" 
COMMISSION_PER_LOT = 6.0 
RR_RATIO = 1.25 
CONFIG = {"EURUSD": {"wrn_step": 0.0010, "pip_size": 0.0001, "ny": (16, 24)}}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')

print(f"=== Сбор данных для ML (Символ: {SYMBOL}) ===")
time_offset_sql = "INTERVAL '1 hour'" 
start_hour, end_hour = cfg["ny"]

# В этом запросе мы собираем ВСЕ возможные признаки (MoE + Triggers + Контекст)
query_dataset = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, avg(ask - bid) AS avg_spread
    FROM read_parquet('{base_path}') GROUP BY bar_time
),
-- 4H контекст
htf_4h AS (
    SELECT time_bucket(INTERVAL '4 hours', timestamp) AS htf_time,
        arg_min(bid, timestamp) AS open_4h, max(bid) AS high_4h, min(bid) AS low_4h, arg_max(bid, timestamp) AS close_4h
    FROM read_parquet('{base_path}') GROUP BY htf_time
),
joined_htf AS (
    SELECT l.*, h.open_4h, h.high_4h, h.low_4h, h.close_4h
    FROM base_data l
    LEFT JOIN htf_4h h ON l.bar_time >= h.htf_time AND l.bar_time < h.htf_time + INTERVAL '4 hours'
),
weekly_range AS (
    SELECT date_trunc('week', bar_time) AS week_start, max(high) AS prev_high, min(low) AS prev_low FROM base_data GROUP BY week_start
),
wrn_levels AS (
    SELECT b.bar_time, CAST(t.level_int AS DOUBLE) * {cfg['wrn_step']} AS wrn_level
    FROM base_data b
    JOIN weekly_range w ON date_trunc('week', b.bar_time) = w.week_start + INTERVAL '7 days'
    CROSS JOIN UNNEST(generate_series(CAST(floor(w.prev_low / {cfg['wrn_step']}) AS BIGINT), CAST(ceil(w.prev_high / {cfg['wrn_step']}) AS BIGINT), 1)) AS t(level_int)
),
with_prev AS (
    SELECT *, 
        LAG(open) OVER (ORDER BY bar_time) AS open_1, LAG(close) OVER (ORDER BY bar_time) AS close_1,
        LAG(open, 2) OVER (ORDER BY bar_time) AS open_2, LAG(close, 2) OVER (ORDER BY bar_time) AS close_2,
        LAG(low, 2) OVER (ORDER BY bar_time) AS low_2, LAG(high, 2) OVER (ORDER BY bar_time) AS high_2,
        LAG(low, 3) OVER (ORDER BY bar_time) AS low_3, LAG(high, 3) OVER (ORDER BY bar_time) AS high_3
    FROM joined_htf
)
SELECT 
    p.bar_time AS signal_time,
    p.close AS entry_price,
    p.avg_spread,
    p.open_4h, p.close_4h, p.high_4h, p.low_4h,
    p.low_2, p.high_2, p.low_3, p.high_3,
    w.wrn_level,
    -- Признак: Bullish FVG (4H) - цена в зоне имбаланса
    CASE WHEN p.low > p.high_3 THEN 1 ELSE 0 END AS in_bullish_fvg_4h,
    -- Признаки паттернов
    CASE WHEN (p.close_2 < p.open_2) AND (p.close > p.open) AND (p.close > (p.open_2+p.close_2)/2) THEN 1 ELSE 0 END AS is_morning_star,
    CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS is_bullish_engulfing,
    CASE WHEN (p.close > p.high_2) THEN 1 ELSE 0 END AS is_shift
FROM with_prev p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.low_2 <= w.wrn_level AND p.close > w.wrn_level
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= {start_hour}
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < {end_hour}
ORDER BY signal_time;
"""

df = con.execute(query_dataset).fetchdf()
print(f"Собрано потенциальных сетапов: {len(df):,}")

# Загружаем свечи для симуляции исхода
bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time, arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# Симуляция (создаем Target: 1 = WIN, 0 = LOSS)
targets = []
pip_size = cfg['pip_size']

for idx, row in df.iterrows():
    entry = row['entry_price']
    sl = row['low_3'] - (2 * pip_size) # Стоп за минимум 3-х свечей
    sl_dist = entry - sl
    
    if sl_dist < 3 * pip_size or sl_dist > 50 * pip_size:
        targets.append(np.nan) # Невалидный сетап
        continue
        
    tp = entry + (sl_dist * RR_RATIO)
    
    future_bars = bars_df.loc[row['signal_time']:].iloc[1:60]
    result = np.nan
    for t, b in future_bars.iterrows():
        if b['low'] <= sl: result = 0; break
        if b['high'] >= tp: result = 1; break
    targets.append(result)

df['target'] = targets
df = df.dropna(subset=['target']) # Убираем незавершенные
print(f"Валидных сделок для обучения: {len(df):,}")

# =====================================================================
# ОБУЧЕНИЕ МОДЕЛИ
# =====================================================================
features = ['avg_spread', 'in_bullish_fvg_4h', 'is_morning_star', 'is_bullish_engulfing', 'is_shift']
X = df[features]
y = df['target'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n--- РЕЗУЛЬТАТЫ ML-МОДЕЛИ ---")
print(f"Точность (Accuracy) на тесте: {accuracy_score(y_test, y_pred):.2%}")
print("\nВажность признаков (Что влияет на прибыль больше всего):")
for feat, imp in zip(features, clf.feature_importances_):
    print(f"  {feat}: {imp:.3f}")

# Какой винрейт у отфильтрованных моделью сделок?
df_test = df.iloc[-len(y_test):].copy()
df_test['prediction'] = y_pred
df_test = df_test[df_test['prediction'] == 1]

if len(df_test) > 0:
    ml_winrate = (df_test['target'].sum() / len(df_test)) * 100
    print(f"\nВинрейт сделок, которые ML-модель одобрила: {ml_winrate:.2f}% (Одобрено {len(df_test)} из {len(y_test)})")
else:
    print("\nМодель отклонила все сделки на тесте.")
