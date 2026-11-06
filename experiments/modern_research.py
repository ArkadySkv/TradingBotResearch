import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from barrier_method import BarrierMethod
from purgedcv import CombinatorialPurgedCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =====================================================================
print("1. Loading and aggregating data...")
SYMBOL = "EURUSD"
BASE_PATH = Path(f"data/parquet/{SYMBOL}/*.parquet")

con = duckdb.connect()
query = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '15 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
        avg(ask - bid) AS avg_spread
    FROM read_parquet('{BASE_PATH}') GROUP BY bar_time
)
SELECT * FROM base_data ORDER BY bar_time;
"""
df = con.execute(query).fetchdf()
df.set_index('bar_time', inplace=True)

df['tr'] = np.maximum(df['high'] - df['low'], 
                      np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                 abs(df['low'] - df['close'].shift(1))))
df['atr'] = df['tr'].rolling(14).mean()
df['returns'] = df['close'].pct_change()
df.dropna(inplace=True)
print(f"Loaded {len(df)} candles.")

# =====================================================================
# 2. TRIPLE-BARRIER (n_barriers=1 для чистых меток)
# =====================================================================
print("\n2. Applying Triple-Barrier Method...")
bm = BarrierMethod(
    returns=df['returns'],
    n=96,
    barrier=0.0015,
    n_barriers=1,      # <-- ИСПРАВЛЕНО: один верхний и один нижний барьер
    center=True
)
labels = bm.labels

# Бинарная разметка: 1 = тейк-профит, 0 = стоп или истечение времени
y = (labels == 1).astype(int).values

df['spread_pips'] = df['avg_spread'] / 0.0001
df['atr_pips'] = df['atr'] / 0.0001
X = df[['spread_pips', 'atr_pips']].values

# Время окончания сделки: приблизительно + n свечей
t1 = df.index + pd.Timedelta(minutes=15 * 96)

print(f"Label distribution (binary): {pd.Series(y).value_counts().to_dict()}")

# =====================================================================
# 3. COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
# =====================================================================
print("\n3. Running Combinatorial Purged CV...")
# <-- ИСПРАВЛЕНО: n_splits и n_test_groups вместо n_folds и n_test_folds
cv = CombinatorialPurgedCV(
    n_splits=6,           # Аналог n_folds
    n_test_groups=2,      # Аналог n_test_folds
    prediction_times=df.index,
    evaluation_times=t1,
    purge_horizon="1D",
    embargo="12H"
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
    eval_metric='logloss'
)
scores = []

# В purgedcv split может не принимать дополнительные аргументы, 
# так как все временные метки заданы в конструкторе.
for train_idx, test_idx in cv.split(X):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

print(f"\nCPCV Results across {len(scores)} backtest paths:")
print(f"Mean Accuracy: {np.mean(scores):.4f}")
print(f"Std Accuracy:  {np.std(scores):.4f}")
print(f"Min/Max:       {np.min(scores):.4f} / {np.max(scores):.4f}")

# =====================================================================
# 4. ВЕРДИКТ
# =====================================================================
print("\n--- FINAL VERDICT ---")
if np.mean(scores) > 0.55:
    print("GO: Model shows a persistent edge across CPCV paths.")
else:
    print("NO-GO: Model performance is not statistically significant.")