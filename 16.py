import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================================
# НАСТРОЙКИ (Возвращаемся на 5M и используем 1D-CNN)
# =====================================================================
SYMBOL = "EURUSD" 
CONFIG = {"EURUSD": {"pip_size": 0.0001, "wrn_step": 0.0010}}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')

print(f"=== 1D-CNN (Визуальный паттерн-реконзинг) для {SYMBOL} ===")
time_offset_sql = "INTERVAL '1 hour'" 

# Собираем данные (только 5M, только ваши WRN зоны)
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
    SELECT *, LAG(low, 2) OVER (ORDER BY bar_time) AS low_2, LAG(close, 2) OVER (ORDER BY bar_time) AS close_2
    FROM base_data
)
SELECT 
    p.bar_time AS signal_time,
    p.close AS entry_price,
    p.avg_spread,
    p.low_2, p.low
FROM with_prev p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.low_2 <= w.wrn_level AND p.close > w.wrn_level
ORDER BY signal_time;
"""
signals = con.execute(query_dataset).fetchdf()
print(f"Собрано сетапов: {len(signals):,}")

# Загружаем 5M свечи
bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# Создаем датасет (Окно из 12 свечей = 1 час)
WINDOW_SIZE = 12
X_data = []
y_data = []
pip_size = cfg['pip_size']

for idx, row in signals.iterrows():
    signal_time = row['signal_time']
    entry = row['entry_price']
    sl = row['low'] - (2 * pip_size) 
    sl_dist = entry - sl
    
    if sl_dist < 3 * pip_size or sl_dist > 50 * pip_size: continue
    tp = entry + (sl_dist * 1.25)
    
    # 1. Считаем TARGET (Win/Loss)
    future_bars = bars_df.loc[signal_time:].iloc[1:60]
    result = np.nan
    for t, b in future_bars.iterrows():
        if b['low'] <= sl: result = 0; break
        if b['high'] >= tp: result = 1; break
        
    if np.isnan(result): continue
        
    # 2. Собираем ОКНО (визуальный образ) 12 свечей ДО сигнала
    past_bars = bars_df.loc[:signal_time].iloc[-WINDOW_SIZE:]
    if len(past_bars) < WINDOW_SIZE: continue
    
    # Нормализуем цены относительно цены входа (чтобы нейросеть видела только форму)
    window_features = []
    for _, bar in past_bars.iterrows():
        window_features.append([
            (bar['open'] - entry) / pip_size,
            (bar['high'] - entry) / pip_size,
            (bar['low'] - entry) / pip_size,
            (bar['close'] - entry) / pip_size
        ])
    
    X_data.append(window_features)
    y_data.append(result)

X_data = np.array(X_data)
y_data = np.array(y_data)
print(f"Сформирован тензор данных для CNN: {X_data.shape}")

# Разделение на Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42, shuffle=False)

# Конвертация в тензоры PyTorch (N, Channels, SequenceLength)
X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# =====================================================================
# АРХИТЕКТУРА НЕЙРОСЕТИ (1D-CNN)
# =====================================================================
class TradingCNN(nn.Module):
    def __init__(self):
        super(TradingCNN, self).__init__()
        # Слой 1: Ищет простые паттерны (например, хвосты свечей)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Слой 2: Ищет сложные паттерны (например, Morning Star)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 3, 64) # 32 канала * 3 (после пулинга)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = TradingCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
epochs = 10
batch_size = 64
dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("\nОбучение нейросети...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

# Тестирование
model.eval()
with torch.no_grad():
    y_prob = model(X_test_t).numpy().flatten()

print("\n--- Результаты 1D-CNN (Визуальное распознавание) ---")
for threshold in [0.50, 0.55, 0.60, 0.65]:
    high_conf_idx = y_prob >= threshold
    if sum(high_conf_idx) > 0:
        actual_winrate = y_test[high_conf_idx].mean()
        print(f"Порог {threshold:.2f} | Сделок: {sum(high_conf_idx):4d} | Винрейт: {actual_winrate:.2%}")
