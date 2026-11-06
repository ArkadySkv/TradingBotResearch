import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape
from gtda.pipeline import make_pipeline

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
SYMBOL = "EURUSD" 
CONFIG = {"EURUSD": {"pip_size": 0.0001, "wrn_step": 0.0010}}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')
time_offset_sql = "INTERVAL '1 hour'" 

print(f"=== TDA (Persistent Homology) для {SYMBOL} ===")

# Запрос (тот же)
query_dataset = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
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
    CROSS JOIN UNNEST(generate_series(CAST(floor(w.prev_low / {cfg['wrn_step']}) AS BIGINT), CAST(ceil(w.prev_high / {cfg['wrn_step']}) AS BIGINT), 1)) AS t(level_int)
),
with_prev AS (
    SELECT *, LAG(low, 2) OVER (ORDER BY bar_time) AS low_2
    FROM base_data
)
SELECT 
    p.bar_time AS signal_time, p.close AS entry_price, p.avg_spread, p.low_2, p.low,
    EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) AS hour_of_day
FROM with_prev p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.low_2 <= w.wrn_level AND p.close > w.wrn_level
ORDER BY signal_time;
"""
signals = con.execute(query_dataset).fetchdf()
signals = signals.dropna()

bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# 72 часа = 864 свечи. Но для TDA мы возьмем 144 свечи (12 часов) для начала (TDA тяжелая).
WINDOW_SIZE = 144 
X_tda, X_context, y_data = [], [], []
pip_size = cfg['pip_size']

print(f"Формирование TDA окон по {WINDOW_SIZE} свечей...")

# Инициализация TDA
persistence = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=-1)
landscape = PersistenceLandscape(n_bins=10, n_layers=2)
tda_pipeline = make_pipeline(persistence, landscape)

count = 0
for idx, row in signals.iterrows():
    signal_time = row['signal_time']
    entry = row['entry_price']
    sl = row['low'] - (2 * pip_size) 
    sl_dist = entry - sl
    if sl_dist < 3 * pip_size or sl_dist > 50 * pip_size: continue
    tp = entry + (sl_dist * 1.25)
    
    future_bars = bars_df.loc[signal_time:].iloc[1:60]
    result = np.nan
    for t, b in future_bars.iterrows():
        if b['low'] <= sl: result = 0; break
        if b['high'] >= tp: result = 1; break
    if np.isnan(result): continue
        
    past_bars = bars_df.loc[:signal_time].iloc[-WINDOW_SIZE:]
    if len(past_bars) < WINDOW_SIZE: continue
    
    # Формируем облако точек: (Время, Нормализованная цена)
    point_cloud = np.column_stack((
        np.arange(WINDOW_SIZE), 
        (past_bars['close'].values - entry) / pip_size
    ))
    
    X_tda.append(point_cloud)
    X_context.append([row['avg_spread'] / pip_size, row['hour_of_day'] / 24.0])
    y_data.append(result)
    
    count += 1
    if count % 1000 == 0:
        print(f"Обработано {count} окон...")
    if count >= 5000: # Ограничим до 5000 для скорости TDA
        break

X_tda = np.array(X_tda)
X_context = np.array(X_context)
y_data = np.array(y_data)

if len(X_tda) == 0:
    print("ОШИБКА: Нет данных для TDA.")
    exit()

print("Вычисление топологических признаков (Persistent Homology)...")
# TDA трансформация (может занять пару минут)
X_tda_features = tda_pipeline.fit_transform(X_tda)
X_tda_flat = X_tda_features.reshape(X_tda_features.shape[0], -1)

print(f"Размерность TDA-признаков: {X_tda_flat.shape}")

# Объединяем TDA и контекст
X_final = np.hstack((X_tda_flat, X_context))

idx_train, idx_test = train_test_split(np.arange(len(y_data)), test_size=0.3, random_state=42, shuffle=False)

X_train, X_test = X_final[idx_train], X_final[idx_test]
y_train, y_test = y_data[idx_train], y_data[idx_test]

clf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

y_prob = clf.predict_proba(X_test)[:, 1]

print("\n--- РЕЗУЛЬТАТЫ TDA ---")
for threshold in [0.50, 0.55, 0.60]:
    high_conf_idx = y_prob >= threshold
    if sum(high_conf_idx) > 0:
        actual_winrate = y_test[high_conf_idx].mean()
        print(f"Порог {threshold:.2f} | Сделок: {sum(high_conf_idx):4d} | Винрейт: {actual_winrate:.2%}")