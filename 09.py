import duckdb
import pandas as pd
from pathlib import Path

# =====================================================================
# НАСТРОЙКИ (Итерация №6: WRN + DO + Ретест + Новостной фильтр)
# =====================================================================
SYMBOL = "EURUSD" 
COMMISSION_PER_LOT = 6.0 
RR_RATIO = 1.25 

CONFIG = {
    "EURUSD": {"wrn_step": 0.0010, "pip_size": 0.0001, "ny": (16, 24)},
}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')

print(f"=== Бэктест {SYMBOL} (WRN + DO + Ретест) ===")

time_offset_sql = "INTERVAL '1 hour'" 
start_hour, end_hour = cfg["ny"]

# Шаг 1: Улучшенный SQL с фильтром DO и проверкой ретеста
query_signals = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, avg(ask - bid) AS avg_spread
    FROM read_parquet('{base_path}') GROUP BY bar_time
),
-- Считаем DO (Открытие Нью-Йоркской сессии)
session_opens AS (
    SELECT 
        date_trunc('day', bar_time + {time_offset_sql}) AS adj_day,
        arg_min(open, bar_time) AS do_level
    FROM base_data
    WHERE EXTRACT(hour FROM (bar_time + {time_offset_sql})) = {start_hour}
    GROUP BY adj_day
),
weekly_range AS (
    SELECT date_trunc('week', bar_time) AS week_start, max(high) AS prev_high, min(low) AS prev_low 
    FROM base_data GROUP BY week_start
),
wrn_levels AS (
    SELECT b.bar_time, CAST(t.level_int AS DOUBLE) * {cfg['wrn_step']} AS wrn_level
    FROM base_data b
    JOIN weekly_range w ON date_trunc('week', b.bar_time) = w.week_start + INTERVAL '7 days'
    CROSS JOIN UNNEST(generate_series(CAST(floor(w.prev_low / {cfg['wrn_step']}) AS BIGINT), CAST(ceil(w.prev_high / {cfg['wrn_step']}) AS BIGINT), 1)) AS t(level_int)
),
with_prev AS (
    SELECT *, 
        LAG(open) OVER (ORDER BY bar_time) AS open_1, 
        LAG(close) OVER (ORDER BY bar_time) AS close_1,
        LAG(high) OVER (ORDER BY bar_time) AS high_1,
        LAG(low) OVER (ORDER BY bar_time) AS low_1,
        LAG(open, 2) OVER (ORDER BY bar_time) AS open_2, 
        LAG(close, 2) OVER (ORDER BY bar_time) AS close_2,
        LAG(low, 2) OVER (ORDER BY bar_time) AS low_2
    FROM base_data
),
pattern_ms AS (
    SELECT *,
        (close_2 < open_2) 
        AND (ABS(close_1 - open_1) < (open_2 - close_2) * 0.3) 
        AND (close > open) 
        AND (close > (open_2 + close_2) / 2) AS is_ms
    FROM with_prev
)
SELECT 
    p.bar_time AS signal_time, 
    p.high_1 AS limit_entry_price,
    LEAST(p.low_2, p.low_1, p.low) AS pattern_low,
    p.avg_spread,
    w.wrn_level,
    s.do_level
FROM pattern_ms p
JOIN wrn_levels w ON p.bar_time = w.bar_time
JOIN session_opens s ON date_trunc('day', p.bar_time + {time_offset_sql}) = s.adj_day
WHERE p.is_ms = TRUE 
  -- Контекст 1: RAID произошел на WRN ИЛИ на DO
  AND (p.low_2 <= w.wrn_level OR p.low_2 <= s.do_level)
  -- Контекст 2: Цена закрылась выше уровня
  AND (p.close > w.wrn_level OR p.close > s.do_level)
  -- Временное окно
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= {start_hour}
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < {end_hour}
  -- НОВОСТНОЙ ФИЛЬТР: Не торгуем в пятницу после 17:00 по GMT+4 (время NFP и закрытия рынка)
  AND NOT (EXTRACT(dow FROM p.bar_time + {time_offset_sql}) = 5 AND EXTRACT(hour FROM p.bar_time + {time_offset_sql}) >= 17)
ORDER BY signal_time;
"""
signals_df = con.execute(query_signals).fetchdf()
print(f"Найдено сигналов (WRN + DO + Фильтры): {len(signals_df):,}")

if len(signals_df) == 0:
    print("Нет сигналов.")
    exit()

# Шаг 2: Загружаем свечи
bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# Шаг 3: Симуляция
wins, losses, cancelled, unresolved = 0, 0, 0, 0
total_pnl_usd = 0
pip_size = cfg['pip_size']

print("Симуляция сделок...")
for idx, row in signals_df.iterrows():
    signal_time = row['signal_time']
    
    limit_price = row['limit_entry_price'] + row['avg_spread']
    sl_price = row['pattern_low'] - (2 * pip_size)
    
    sl_dist = limit_price - sl_price
    if sl_dist < 3 * pip_size or sl_dist > 50 * pip_size:
        continue 
        
    tp_dist = sl_dist * RR_RATIO
    tp_price = limit_price + tp_dist
    
    future_bars = bars_df.loc[signal_time:].iloc[1:13] 
    
    entry_time = None
    for t, b in future_bars.iterrows():
        if b['low'] <= limit_price: 
            entry_time = t
            break
            
    if entry_time is None:
        cancelled += 1
        continue 
        
    trade_bars = bars_df.loc[entry_time:].iloc[1:60]
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
        total_pnl_usd += (tp_dist / pip_size) * 10.0
    elif trade_result == "LOSS":
        losses += 1
        total_pnl_usd -= (sl_dist / pip_size) * 10.0
    else:
        unresolved += 1

# Шаг 4: Итоги
total_trades = wins + losses
if total_trades == 0:
    print("Ни одна сделка не закрылась.")
    exit()

win_rate = (wins / total_trades) * 100
commission = total_trades * COMMISSION_PER_LOT
net_pnl_usd = total_pnl_usd - commission

print(f"\n--- РЕЗУЛЬТАТЫ БЭКТЕСТА (WRN + DO) ---")
print(f"Всего сигналов: {len(signals_df)}")
print(f"Отменено (нет ретеста): {cancelled}")
print(f"Закрытых сделок: {total_trades}")
print(f"Винрейт: {win_rate:.2f}%")
print(f"Чистый PnL: ${net_pnl_usd:.2f}")
