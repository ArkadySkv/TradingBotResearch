import duckdb
import pandas as pd
from pathlib import Path

# =====================================================================
# НАСТРОЙКИ (Итерация №3: ATR + Вход на ретесте)
# =====================================================================
SYMBOL = "EURUSD" 
ATR_PERIOD = 14
SL_ATR_MULT = 1.5   # Стоп = 1.5 * ATR
TP_ATR_MULT = 1.875 # Тейк = 1.875 * ATR (соотношение 1:1.25)
COMMISSION_PER_LOT = 6.0 

CONFIG = {
    "EURUSD": {"wrn_step": 0.0010, "pip_size": 0.0001, "ny": (16, 24)},
}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')

print(f"=== Бэктест {SYMBOL} (ATR + Limit Entry) ===")

time_offset_sql = "INTERVAL '1 hour'" 
start_hour, end_hour = cfg["ny"]

# Шаг 1: Получаем сигналы + ATR
query_signals = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, avg(ask - bid) AS avg_spread
    FROM read_parquet('{base_path}') GROUP BY bar_time
),
-- Считаем ATR (упрощенно через TR)
with_atr AS (
    SELECT *,
        GREATEST(
            high - low,
            ABS(high - LAG(close) OVER (ORDER BY bar_time)),
            ABS(low - LAG(close) OVER (ORDER BY bar_time))
        ) AS tr
    FROM base_data
),
with_avg_atr AS (
    SELECT *,
        AVG(tr) OVER (ORDER BY bar_time ROWS BETWEEN {ATR_PERIOD-1} PRECEDING AND CURRENT ROW) AS atr
    FROM with_atr
),
weekly_range AS (
    SELECT date_trunc('week', bar_time) AS week_start, max(high) AS prev_high, min(low) AS prev_low FROM with_avg_atr GROUP BY week_start
),
wrn_levels AS (
    SELECT b.bar_time, CAST(t.level_int AS DOUBLE) * {cfg['wrn_step']} AS wrn_level
    FROM with_avg_atr b
    JOIN weekly_range w ON date_trunc('week', b.bar_time) = w.week_start + INTERVAL '7 days'
    CROSS JOIN UNNEST(generate_series(CAST(floor(w.prev_low / {cfg['wrn_step']}) AS BIGINT), CAST(ceil(w.prev_high / {cfg['wrn_step']}) AS BIGINT), 1)) AS t(level_int)
),
with_prev AS (
    SELECT *, LAG(open) OVER (ORDER BY bar_time) AS open_1, LAG(close) OVER (ORDER BY bar_time) AS close_1,
        LAG(open, 2) OVER (ORDER BY bar_time) AS open_2, LAG(close, 2) OVER (ORDER BY bar_time) AS close_2, LAG(low, 2) OVER (ORDER BY bar_time) AS low_2
    FROM with_avg_atr
),
pattern_ms AS (
    SELECT *, (close_2 < open_2) AND (ABS(close_1 - open_1) < (open_2 - close_2) * 0.3) AND (close > open) AND (close > (open_2 + close_2) / 2) AS is_ms
    FROM with_prev
)
SELECT p.bar_time AS signal_time, p.close AS signal_close, p.avg_spread, p.atr, w.wrn_level AS level_price
FROM pattern_ms p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.is_ms = TRUE AND p.low_2 <= w.wrn_level AND p.close > w.wrn_level
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= {start_hour}
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < {end_hour}
ORDER BY signal_time;
"""
signals_df = con.execute(query_signals).fetchdf()
print(f"Найдено сигналов (с ATR): {len(signals_df):,}")

if len(signals_df) == 0:
    print("Нет сигналов.")
    exit()

# Шаг 2: Загружаем свечи для симуляции
bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# Шаг 3: Симуляция (Лимитный вход на 50% откате тела звезды)
wins, losses, unresolved = 0, 0, 0
total_pnl_pips = 0

print("Симуляция сделок...")
for idx, row in signals_df.iterrows():
    signal_time = row['signal_time']
    atr = row['atr']
    
    # Мы хотим купить на откате. 
    # Цена закрытия была signal_close. Откат = signal_close - (50% от размера свечи)
    # Но для простоты: мы ставим лимитку на 2 пипса ниже закрытия сигнальной свечи.
    limit_price = row['signal_close'] - (2 * cfg['pip_size']) + row['avg_spread']
    
    future_bars = bars_df.loc[signal_time:].iloc[1:]
    
    entry_time = None
    # Ждем, пока цена не коснется нашей лимитки (Low <= limit_price)
    for t, b in future_bars.head(12).iterrows(): # Ждем максимум 1 час (12 свечей)
        if b['low'] <= limit_price:
            entry_time = t
            break
            
    if entry_time is None:
        continue # Цена не откатилась, сделка не открылась
        
    entry_price = limit_price
    # Динамические SL и TP на основе ATR
    sl_dist = atr * SL_ATR_MULT
    tp_dist = atr * TP_ATR_MULT
    
    sl_price = entry_price - sl_dist
    tp_price = entry_price + tp_dist
    
    # Симуляция после входа
    trade_bars = bars_df.loc[entry_time:].iloc[1:100]
    trade_result = None
    
    for t, b in trade_bars.iterrows():
        if b['low'] <= sl_price:
            trade_result = "LOSS"
            break
        if b['high'] >= tp_price:
            trade_result = "WIN"
            break
            
    if trade_result == "WIN":
        wins += 1
        total_pnl_pips += (tp_dist / cfg['pip_size'])
    elif trade_result == "LOSS":
        losses += 1
        total_pnl_pips -= (sl_dist / cfg['pip_size'])
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

print(f"\n--- РЕЗУЛЬТАТЫ БЭКТЕСТА (ATR + Limit) ---")
print(f"Закрытых сделок: {total_trades}")
print(f"Выигрышей (WIN): {wins}")
print(f"Проигрышей (LOSS): {losses}")
print(f"Винрейт: {win_rate:.2f}%")
print(f"Чистый PnL: ${net_pnl_usd:.2f}")
