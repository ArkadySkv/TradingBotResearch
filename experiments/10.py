import duckdb
import pandas as pd
from pathlib import Path

# =====================================================================
# НАСТРОЙКИ (Итерация №7: MS/ES + Shift + Отдельные WRN/DO)
# =====================================================================
SYMBOL = "EURUSD" 
COMMISSION_PER_LOT = 6.0 
RR_RATIO = 1.25 
MAX_SPREAD_PIPS = 2.5 # Фильтр новостей: не входим, если спред расширен

CONFIG = {
    "EURUSD": {"wrn_step": 0.0010, "pip_size": 0.0001, "ny": (16, 24)},
}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')

print(f"=== Бэктест {SYMBOL} (Логика Shift + Long/Short) ===")

time_offset_sql = "INTERVAL '1 hour'" 
start_hour, end_hour = cfg["ny"]

query_signals = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, avg(ask - bid) AS avg_spread
    FROM read_parquet('{base_path}') GROUP BY bar_time
),
-- DO (Открытие сессии)
session_opens AS (
    SELECT 
        date_trunc('day', bar_time + {time_offset_sql}) AS adj_day,
        arg_min(open, bar_time) AS do_level,
        min(bar_time) AS do_time
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
        LAG(high, 2) OVER (ORDER BY bar_time) AS high_2,
        LAG(low, 2) OVER (ORDER BY bar_time) AS low_2
    FROM base_data
),
-- 1. Логика Morning Star + Shift (Слом вверх)
pattern_long AS (
    SELECT *,
        -- MS
        (close_2 < open_2) 
        AND (ABS(close_1 - open_1) < ABS(open_2 - close_2) * 0.5) 
        AND (close > open) 
        AND (close > (open_2 + close_2) / 2)
        -- SHIFT (Пробой максимума свечи свипа)
        AND (close > high_2) AS is_valid_long
    FROM with_prev
),
-- 2. Логика Evening Star + Shift (Слом вниз)
pattern_short AS (
    SELECT *,
        -- ES
        (close_2 > open_2) 
        AND (ABS(close_1 - open_1) < ABS(open_2 - close_2) * 0.5) 
        AND (close < open) 
        AND (close < (open_2 + close_2) / 2)
        -- SHIFT (Пробой минимума свечи свипа)
        AND (close < low_2) AS is_valid_short
    FROM with_prev
)
-- LONG SIGNALS (WRN + DO)
SELECT p.bar_time AS signal_time, p.high_1 AS limit_entry, LEAST(p.low_2, p.low_1, p.low) AS sl_extreme, p.avg_spread, 'LONG' as direction
FROM pattern_long p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.is_valid_long = TRUE AND p.low_2 <= w.wrn_level AND p.close > w.wrn_level
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= {start_hour}
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < {end_hour}

UNION ALL

SELECT p.bar_time AS signal_time, p.high_1 AS limit_entry, LEAST(p.low_2, p.low_1, p.low) AS sl_extreme, p.avg_spread, 'LONG' as direction
FROM pattern_long p
JOIN session_opens s ON date_trunc('day', p.bar_time + {time_offset_sql}) = s.adj_day
WHERE p.is_valid_long = TRUE AND p.low_2 <= s.do_level AND p.close > s.do_level AND p.bar_time > s.do_time
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= {start_hour}
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < {end_hour}

UNION ALL

-- SHORT SIGNALS (WRN + DO)
SELECT p.bar_time AS signal_time, p.low_1 AS limit_entry, GREATEST(p.high_2, p.high_1, p.high) AS sl_extreme, p.avg_spread, 'SHORT' as direction
FROM pattern_short p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.is_valid_short = TRUE AND p.high_2 >= w.wrn_level AND p.close < w.wrn_level
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= {start_hour}
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < {end_hour}

UNION ALL

SELECT p.bar_time AS signal_time, p.low_1 AS limit_entry, GREATEST(p.high_2, p.high_1, p.high) AS sl_extreme, p.avg_spread, 'SHORT' as direction
FROM pattern_short p
JOIN session_opens s ON date_trunc('day', p.bar_time + {time_offset_sql}) = s.adj_day
WHERE p.is_valid_short = TRUE AND p.high_2 >= s.do_level AND p.close < s.do_level AND p.bar_time > s.do_time
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= {start_hour}
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < {end_hour}
ORDER BY signal_time;
"""

signals_df = con.execute(query_signals).fetchdf()
print(f"Найдено сигналов (Long/Short + Shift): {len(signals_df):,}")

if len(signals_df) == 0:
    print("Нет сигналов.")
    exit()

# Загружаем свечи
bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# Симуляция
wins, losses, cancelled = 0, 0, 0
total_pnl_usd = 0
pip_size = cfg['pip_size']

for idx, row in signals_df.iterrows():
    # Фильтр широкого спреда (новости)
    if row['avg_spread'] / pip_size > MAX_SPREAD_PIPS:
        continue
        
    signal_time = row['signal_time']
    limit_price = row['limit_entry'] + (row['avg_spread'] if row['direction'] == 'LONG' else -row['avg_spread'])
    
    if row['direction'] == 'LONG':
        sl_price = row['sl_extreme'] - (2 * pip_size)
        sl_dist = limit_price - sl_price
        tp_price = limit_price + (sl_dist * RR_RATIO)
    else:
        sl_price = row['sl_extreme'] + (2 * pip_size)
        sl_dist = sl_price - limit_price
        tp_price = limit_price - (sl_dist * RR_RATIO)
        
    if sl_dist < 3 * pip_size or sl_dist > 50 * pip_size:
        continue
        
    # Ждем ретеста
    future_bars = bars_df.loc[signal_time:].iloc[1:13] 
    entry_time = None
    for t, b in future_bars.iterrows():
        if row['direction'] == 'LONG' and b['low'] <= limit_price: 
            entry_time = t; break
        if row['direction'] == 'SHORT' and b['high'] >= limit_price:
            entry_time = t; break
            
    if entry_time is None:
        cancelled += 1
        continue 
        
    # Симуляция после входа
    trade_bars = bars_df.loc[entry_time:].iloc[1:60]
    trade_result = None
    
    for t, b in trade_bars.iterrows():
        if row['direction'] == 'LONG':
            if b['low'] <= sl_price: trade_result = "LOSS"; break
            if b['high'] >= tp_price: trade_result = "WIN"; break
        else:
            if b['high'] >= sl_price: trade_result = "LOSS"; break
            if b['low'] <= tp_price: trade_result = "WIN"; break
            
    if trade_result == "WIN":
        wins += 1
        total_pnl_usd += (sl_dist * RR_RATIO / pip_size) * 10.0
    elif trade_result == "LOSS":
        losses += 1
        total_pnl_usd -= (sl_dist / pip_size) * 10.0

total_trades = wins + losses
if total_trades == 0:
    print("Ни одна сделка не закрылась.")
    exit()

win_rate = (wins / total_trades) * 100
net_pnl_usd = total_pnl_usd - (total_trades * COMMISSION_PER_LOT)

print(f"\n--- РЕЗУЛЬТАТЫ (Long/Short + Shift) ---")
print(f"Закрытых сделок: {total_trades}")
print(f"Выигрышей: {wins} | Проигрышей: {losses}")
print(f"Винрейт: {win_rate:.2f}%")
print(f"Чистый PnL: ${net_pnl_usd:.2f}")
