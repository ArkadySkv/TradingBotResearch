import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================================================
# НАСТРОЙКИ (72 часа = 864 свечи на 5M)
# =====================================================================
SYMBOL = "EURUSD" 
CONFIG = {"EURUSD": {"pip_size": 0.0001, "wrn_step": 0.0010}}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')
time_offset_sql = "INTERVAL '1 hour'" 

print(f"=== CNN + GRU (72 часа) v2 (Fixing NaN) для {SYMBOL} ===")

query_dataset = f"""
WITH raw_ticks AS (
    SELECT timestamp, bid, ask, volume,
        SUM(bid * volume) OVER (PARTITION BY date_trunc('day', timestamp) ORDER BY timestamp) / SUM(volume) OVER (PARTITION BY date_trunc('day', timestamp) ORDER BY timestamp) AS vwap
    FROM read_parquet('{base_path}')
),
base_data AS (
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
        avg(ask - bid) AS avg_spread, arg_max(vwap, timestamp) AS vwap_close
    FROM raw_ticks GROUP BY bar_time
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
htf_4h AS (
    SELECT time_bucket(INTERVAL '4 hours', timestamp) AS htf_time, max(bid) AS high_4h, min(bid) AS low_4h
    FROM read_parquet('{base_path}') GROUP BY htf_time
),
joined_htf AS (
    SELECT l.*, h.high_4h, h.low_4h
    FROM base_data l
    LEFT JOIN htf_4h h ON l.bar_time >= h.htf_time AND l.bar_time < h.htf_time + INTERVAL '4 hours'
),
with_prev AS (
    SELECT *, LAG(low, 2) OVER (ORDER BY bar_time) AS low_2, LAG(high, 3) OVER (ORDER BY bar_time) AS high_3
    FROM joined_htf
)
SELECT 
    p.bar_time AS signal_time, p.close AS entry_price, p.avg_spread, p.low_2, p.low, p.vwap_close,
    EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) AS hour_of_day,
    CASE WHEN p.low > p.high_3 THEN 1 ELSE 0 END AS in_bullish_fvg
FROM with_prev p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.low_2 <= w.wrn_level AND p.close > w.wrn_level
ORDER BY signal_time;
"""
signals = con.execute(query_dataset).fetchdf()
signals = signals.dropna() # Убираем любые NaN из VWAP

bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

WINDOW_SIZE = 72 * 12 # 864 свечи
X_sequence, X_context, y_data = [], [], []
pip_size = cfg['pip_size']

print(f"Формирование окон по {WINDOW_SIZE} свечей...")

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
    
    window_features = []
    for _, bar in past_bars.iterrows():
        window_features.append([
            (bar['open'] - entry) / pip_size,
            (bar['high'] - entry) / pip_size,
            (bar['low'] - entry) / pip_size,
            (bar['close'] - entry) / pip_size
        ])
    
    context_features = [
        row['avg_spread'] / pip_size,
        row['hour_of_day'] / 24.0, 
        row['in_bullish_fvg'],
        (entry - row['vwap_close']) / pip_size 
    ]
    
    X_sequence.append(window_features)
    X_context.append(context_features)
    y_data.append(result)

X_sequence = np.array(X_sequence)
X_context = np.array(X_context)
y_data = np.array(y_data)

# ИСПРАВЛЕНИЕ 1: Нормализация контекста (StandardScaler)
scaler = StandardScaler()
X_context = scaler.fit_transform(X_context)

idx_train, idx_test = train_test_split(np.arange(len(y_data)), test_size=0.3, random_state=42, shuffle=False)

seq_train = torch.tensor(X_sequence[idx_train], dtype=torch.float32).permute(0, 2, 1)
ctx_train = torch.tensor(X_context[idx_train], dtype=torch.float32)
y_train = torch.tensor(y_data[idx_train], dtype=torch.float32).unsqueeze(1)

seq_test = torch.tensor(X_sequence[idx_test], dtype=torch.float32).permute(0, 2, 1)
ctx_test = torch.tensor(X_context[idx_test], dtype=torch.float32)
y_test = y_data[idx_test]

# Упрощенная архитектура, чтобы не взрывалась
class CNN_GRU_TradingNet(nn.Module):
    def __init__(self):
        super(CNN_GRU_TradingNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4) 
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4) 
        
        # 1 слой GRU вместо 2-х (чтобы избежать взрыва)
        self.gru = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True)
        
        combined_features = 16 + 4 
        self.fc1 = nn.Linear(combined_features, 16)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, seq, ctx):
        x = self.pool1(self.relu1(self.conv1(seq)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        
        _, hidden = self.gru(x)
        x = hidden[-1]
        
        combined = torch.cat((x, ctx), dim=1)
        out = self.relu3(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out)

model = CNN_GRU_TradingNet()
pos_weight = torch.tensor([sum(y_train == 0).item() / max(sum(y_train == 1).item(), 1)])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# Уменьшили learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 15
batch_size = 64
dataset = torch.utils.data.TensorDataset(seq_train, ctx_train, y_train)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("\nОбучение CNN+GRU v2...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_seq, batch_ctx, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_seq, batch_ctx)
        loss = criterion(outputs, batch_y)
        
        # Проверка на NaN
        if torch.isnan(loss):
            print("Обнаружен NaN в Loss! Прерываем эпоху.")
            break
            
        loss.backward()
        
        # ИСПРАВЛЕНИЕ 2: Gradient Clipping (обрезаем градиенты)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
    if not np.isnan(total_loss):
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

# Тестирование
model.eval()
with torch.no_grad():
    y_prob = torch.sigmoid(model(seq_test, ctx_test)).numpy().flatten()

print("\n--- РЕЗУЛЬТАТЫ CNN+GRU (72 часа) ---")
for threshold in [0.50, 0.55, 0.60, 0.65]:
    high_conf_idx = y_prob >= threshold
    if sum(high_conf_idx) > 0:
        actual_winrate = y_test[high_conf_idx].mean()
        print(f"Порог {threshold:.2f} | Сделок: {sum(high_conf_idx):4d} | Винрейт: {actual_winrate:.2%}")
