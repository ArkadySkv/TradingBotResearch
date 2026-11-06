import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from barrier_method import BarrierMethod
from purgedcv import CombinatorialPurgedCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import iisignature
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
print(f"Loaded {len(df)} candles.")

# =====================================================================
# 2. СОЗДАНИЕ ПРИЗНАКОВ НА ОСНОВЕ СИГНАТУР ПУТЕЙ
# =====================================================================
print("\n2. Computing Path Signatures...")

# Определяем окно (например, последние 24 часа = 96 свечей по 15М)
WINDOW_SIZE = 96
DEPTH = 3  # Глубина сигнатуры (3 или 4 — разумный баланс)

# Создаем многоканальный путь: [Цена (close), Спред]
# Важно: нормализуем данные, чтобы сигнатура была устойчивой
df['close_norm'] = (df['close'] - df['close'].rolling(WINDOW_SIZE).mean()) / df['close'].rolling(WINDOW_SIZE).std()
df['spread_norm'] = (df['avg_spread'] - df['avg_spread'].rolling(WINDOW_SIZE).mean()) / df['avg_spread'].rolling(WINDOW_SIZE).std()
df.dropna(inplace=True)

# Функция для вычисления сигнатуры
def get_signature(window_data):
    # window_data: shape (time_steps, channels)
    # iisignature ожидает путь как массив float
    return iisignature.sig(window_data, DEPTH)

# Применяем скользящее окно для создания признаков
signature_features = []
for i in range(WINDOW_SIZE, len(df)):
    window = df[['close_norm', 'spread_norm']].iloc[i-WINDOW_SIZE:i].values
    sig = get_signature(window)
    signature_features.append(sig)

# Создаем DataFrame с признаками
# Количество признаков для 2 каналов и глубины 3: d + d^2 + d^3 = 2 + 4 + 8 = 14
feature_names = [f'sig_{i}' for i in range(len(signature_features[0]))]
X_sig = pd.DataFrame(signature_features, columns=feature_names, index=df.index[WINDOW_SIZE:])

print(f"Generated {X_sig.shape[1]} signature features.")

# =====================================================================
# 3. TRIPLE-BARRIER LABELING (на тех же данных)
# =====================================================================
print("\n3. Applying Triple-Barrier Method...")
# Используем только те данные, для которых есть признаки
df_signals = df.iloc[WINDOW_SIZE:].copy()
df_signals['returns'] = df_signals['close'].pct_change()
df_signals.dropna(inplace=True)

bm = BarrierMethod(
    returns=df_signals['returns'],
    n=96,
    barrier=0.0015,
    n_barriers=1,
    center=True
)
labels = bm.labels
y = (labels == 1).astype(int).values

# Синхронизируем X и y
X = X_sig.loc[df_signals.index].values
y = y[:len(X)]  # на случай, если последние метки не определены

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# =====================================================================
# 4. COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
# =====================================================================
print("\n4. Running Combinatorial Purged CV with Signature Features...")

t1 = df_signals.index + pd.Timedelta(minutes=15 * 96)

cv = CombinatorialPurgedCV(
    n_splits=6,
    n_test_groups=2,
    prediction_times=df_signals.index,
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
# 5. ВЕРДИКТ
# =====================================================================
print("\n--- FINAL VERDICT ---")
if np.mean(scores) > 0.55:
    print("GO: Path Signatures provide a persistent edge across CPCV paths.")
else:
    print("NO-GO: Even with Path Signatures, the model shows no statistically significant edge.")