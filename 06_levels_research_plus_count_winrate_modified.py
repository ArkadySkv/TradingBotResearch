import duckdb
import pandas as pd
from pathlib import Path

# =====================================================================
# НАСТРОЙКИ (Итерация №2: С фильтром тренда)
# =====================================================================
SYMBOL = "EURUSD" 
SL_PIPS = 20       
TP_PIPS = 25       
COMMISSION_PER_LOT = 6.0 

CONFIG = {
    "EURUSD": {"wrn_step": 0.0010, "pip_size": 0.0001, "ny": (16, 24)},
    # Можете добавить другие...
}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')

print(f"=== Бэктест {SYMBOL} (С фильтром EMA200) | SL: {SL_PIPS} | TP: {TP_PIPS} ===")

time_offset_sql = "INTERVAL '1 hour'" 
start_hour, end_hour = cfg["ny"]

# Шаг 1: Получаем сигналы + добавляем EMA200
query_signals = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, avg(ask - bid) AS avg_spread
    FROM read_parquet('{base_path}') GROUP BY bar_time
),
-- Считаем EMA 200 на 5-минутках
with_ema AS (
    SELECT *,
        AVG(close) OVER (ORDER BY bar_time ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ema_200
    FROM base_data
),
weekly_range AS (
    SELECT date_trunc('week', bar_time) AS week_start, max(high) AS prev_high, min(low) AS prev_low FROM with_ema GROUP BY week_start
),
wrn_levels AS (
    SELECT b.bar_time, CAST(t.level_int AS DOUBLE) * {cfg['wrn_step']} AS wrn_level
    FROM with_ema b
    JOIN weekly_range w ON date_trunc('week', b.bar_time) = w.week_start + INTERVAL '7 days'
    CROSS JOIN UNNEST(generate_series(CAST(floor(w.prev_low / {cfg['wrn_step']}) AS BIGINT), CAST(ceil(w.prev_high / {cfg['wrn_step']}) AS BIGINT), 1)) AS t(level_int)
),
with_prev AS (
    SELECT *, LAG(open) OVER (ORDER BY bar_time) AS open_1, LAG(close) OVER (ORDER BY bar_time) AS close_1,
        LAG(open, 2) OVER (ORDER BY bar_time) AS open_2, LAG(close, 2) OVER (ORDER BY bar_time) AS close_2, LAG(low, 2) OVER (ORDER BY bar_time) AS low_2
    FROM with_ema
),
pattern_ms AS (
    SELECT *, (close_2 < open_2) AND (ABS(close_1 - open_1) < (open_2 - close_2) * 0.3) AND (close > open) AND (close > (open_2 + close_2) / 2) AS is_ms
    FROM with_prev
)
SELECT p.bar_time AS signal_time, p.close AS signal_close, p.avg_spread, w.wrn_level AS level_price
FROM pattern_ms p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.is_ms = TRUE AND p.low_2 <= w.wrn_level AND p.close > w.wrn_level
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= {start_hour}
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < {end_hour}
  -- ФИЛЬТР ТРЕНДА: Покупаем только если цена ЗАКРЫТИЯ выше EMA200
  AND p.close > p.ema_200
ORDER BY signal_time;
"""
signals_df = con.execute(query_signals).fetchdf()
print(f"Найдено сигналов (с фильтром EMA): {len(signals_df):,}")

if len(signals_df) == 0:
    print("Нет сигналов. Завершаем.")
    exit()

# Шаг 2: Загружаем все свечи
print("Загрузка 5M свечей...")
bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# Шаг 3: Симуляция (Вход по Open СЛЕДУЮЩЕЙ свечи)
pip_size = cfg['pip_size']
sl_dist = SL_PIPS * pip_size
tp_dist = TP_PIPS * pip_size

wins, losses, unresolved = 0, 0, 0
total_pnl_pips = 0

print("Симуляция сделок...")
for idx, row in signals_df.iterrows():
    signal_time = row['signal_time']
    
    # Ищем следующую свечу после сигнала для входа
    future_bars = bars_df.loc[signal_time:].iloc[1:] 
    if len(future_bars) == 0: continue
    
    # Входим по цене Open следующей свечи + спред (покупаем по Ask)
    entry_time = future_bars.index[0]
    entry_price = future_bars.iloc[0]['open'] + row['avg_spread']
    
    tp_price = entry_price + tp_dist
    sl_price = entry_price - sl_dist
    
    # Смотрим на 100 свечей вперед
    trade_bars = future_bars.iloc[1:100]
    trade_result = None
    
    for future_time, future_row in trade_bars.iterrows():
        if future_row['low'] <= sl_price:
            trade_result = "LOSS"
            break
        if future_row['high'] >= tp_price:
            trade_result = "WIN"
            break
            
    if trade_result == "WIN":
        wins += 1
        total_pnl_pips += TP_PIPS
    elif trade_result == "LOSS":
        losses += 1
        total_pnl_pips -= SL_PIPS
    else:
        unresolved += 1

# Шаг 4: Итоги
total_trades = wins + losses
if total_trades == 0:
    print("Ни одна сделка не закрылась.")
    exit()

win_rate = (wins / total_trades) * 100
gross_pnl_usd = total_pnl_pips * 10.0
commission = total_trades * COMMISSION_PER_LOT
net_pnl_usd = gross_pnl_usd - commission

print(f"\n--- РЕЗУЛЬТАТЫ БЭКТЕСТА (С EMA) ---")
print(f"Закрытых сделок: {total_trades}")
print(f"Выигрышей (WIN): {wins}")
print(f"Проигрышей (LOSS): {losses}")
print(f"Винрейт: {win_rate:.2f}%")
print(f"Чистый PnL: ${net_pnl_usd:.2f}")
