import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
SYMBOL = "EURUSD" 
CONFIG = {"EURUSD": {"pip_size": 0.0001, "wrn_step": 0.0010}}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
time_offset_sql = "INTERVAL '1 hour'" # Сдвиг для GMT+4 (если сервер GMT+3)
COMMISSION_PER_LOT = 6.0 
TIMEFRAME = 15
FIXED_RISK_USD = 50.0 
PIP_VALUE_USD = 10.0  

print(f"=== H5.1: 16:00 GMT+4 DO FILTER (15M) ===")

con = duckdb.connect(':memory:')

# SQL: Ищем 1H свечу в 16:00, определяем DO, фильтруем 15M
query_dataset = f"""
WITH base_data_15m AS (
    SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
        avg(ask - bid) AS avg_spread
    FROM read_parquet('{base_path}') GROUP BY bar_time
),
base_data_1h AS (
    SELECT time_bucket(INTERVAL '1 hour', timestamp) AS htf_time,
        arg_min(bid, timestamp) AS open_1h, max(bid) AS high_1h, min(bid) AS low_1h, arg_max(bid, timestamp) AS close_1h
    FROM read_parquet('{base_path}') GROUP BY htf_time
),
-- 1. Ищем 1H свечу в 16:00 GMT+4 и определяем уровень по ВАШЕЙ логике
daily_open_levels AS (
    SELECT 
        date_trunc('day', htf_time + {time_offset_sql}) AS adj_day,
        CASE 
            WHEN close_1h > open_1h THEN high_1h -- Бычья свеча -> ставим на High
            ELSE low_1h                          -- Медвежья свеча -> ставим на Low
        END AS do_level
    FROM base_data_1h
    WHERE EXTRACT(hour FROM (htf_time + {time_offset_sql})) = 16 
      AND EXTRACT(minute FROM (htf_time + {time_offset_sql})) = 0
),
with_atr AS (
    SELECT *, GREATEST(high - low, ABS(high - LAG(close) OVER (ORDER BY bar_time)), ABS(low - LAG(close) OVER (ORDER BY bar_time))) AS tr
    FROM base_data_15m
),
with_avg_atr AS (
    SELECT *, AVG(tr) OVER (ORDER BY bar_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS atr
    FROM with_atr
),
weekly_range AS (
    SELECT date_trunc('week', bar_time) AS week_start, max(high) AS prev_high, min(low) AS prev_low FROM with_avg_atr GROUP BY week_start
),
wrn_levels AS (
    SELECT b.bar_time, CAST(t.level_int AS DOUBLE) * {cfg['wrn_step']} AS wrn_level
    FROM with_avg_atr b
    JOIN weekly_range w ON date_trunc('week', b.bar_time) = w.week_start + INTERVAL '7 days'
    CROSS JOIN UNNEST(generate_series(CAST(floor(w.prev_low / {cfg['wrn_step']}) AS BIGINT) - 1, CAST(ceil(w.prev_high / {cfg['wrn_step']}) AS BIGINT) + 1, 1)) AS t(level_int)
),
with_prev AS (
    SELECT *, 
        LAG(open) OVER (ORDER BY bar_time) AS open_1, LAG(close) OVER (ORDER BY bar_time) AS close_1,
        LAG(high) OVER (ORDER BY bar_time) AS high_1, LAG(low) OVER (ORDER BY bar_time) AS low_1
    FROM with_avg_atr
),
raw_signals AS (
    SELECT 
        p.bar_time AS signal_time, p.close AS entry_price, p.avg_spread, p.low, p.high, p.atr, d.do_level,
        
        -- MoE
        CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS moe_bull_engulf,
        CASE WHEN (p.close_1 > p.open_1) AND (p.close < p.open) AND (p.close < p.open_1) THEN 1 ELSE 0 END AS moe_bear_engulf,
        
        -- RAID
        CASE WHEN p.low_1 <= w.wrn_level AND p.close > w.wrn_level THEN 1 ELSE 0 END AS raid_wrn_long,
        CASE WHEN p.high_1 >= w.wrn_level AND p.close < w.wrn_level THEN 1 ELSE 0 END AS raid_wrn_short,
        
        -- SWEEP + SHIFT
        CASE WHEN p.low_1 <= p.low AND p.close > p.high_1 THEN 1 ELSE 0 END AS sweep_shift_long,
        CASE WHEN p.high_1 >= p.high AND p.close < p.low_1 THEN 1 ELSE 0 END AS sweep_shift_short
    FROM with_prev p
    LEFT JOIN wrn_levels w ON p.bar_time = w.bar_time
    LEFT JOIN daily_open_levels d ON date_trunc('day', p.bar_time + {time_offset_sql}) = d.adj_day
)
SELECT DISTINCT ON (signal_time) * FROM raw_signals 
WHERE (raid_wrn_long = 1 OR raid_wrn_short = 1) 
  AND (moe_bull_engulf=1 OR moe_bear_engulf=1)
  -- ФИЛЬТР ТРЕНДА: Используем ваш уровень DO
  AND (
      (sweep_shift_long = 1 AND entry_price > do_level) 
      OR 
      (sweep_shift_short = 1 AND entry_price < do_level)
  )
ORDER BY signal_time;
"""
signals = con.execute(query_dataset).fetchdf()
print(f"Найдено сетапов после фильтра 16:00 DO: {len(signals)}")

if len(signals) == 0:
    print("Фильтр слишком строгий.")
    exit()

bars_df = con.execute(f"SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time, arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# Бэктест
wins, losses = 0, 0
total_pnl_usd = 0
pip_size = cfg['pip_size']
years_pnl = {}

for idx, row in signals.iterrows():
    entry = row['entry_price']
    signal_idx = bars_df.index.get_loc(row['signal_time'])
    if signal_idx < 1: continue
    
    prev_bar = bars_df.iloc[signal_idx - 1]
    direction = 1 if row['sweep_shift_long'] == 1 else -1
    
    if direction == 1:
        sl = prev_bar['low'] - (2 * pip_size)
        sl_dist = entry - sl
        tp = entry + sl_dist # 1:1
    else:
        sl = prev_bar['high'] + (2 * pip_size)
        sl_dist = sl - entry
        tp = entry - sl_dist
        
    sl_pips = sl_dist / pip_size
    if sl_pips < 5 or sl_pips > 60: continue
    
    future_bars = bars_df.iloc[signal_idx + 1: signal_idx + 100]
    result = None
    for t, b in future_bars.iterrows():
        if direction == 1:
            if b['low'] <= sl: result = 0; break
            if b['high'] >= tp: result = 1; break
        else:
            if b['high'] >= sl: result = 0; break
            if b['low'] <= tp: result = 1; break
            
    if result is None: continue
        
    lot_size = FIXED_RISK_USD / (sl_pips * PIP_VALUE_USD)
    lot_size = max(round(lot_size, 2), 0.01)
    comm = lot_size * COMMISSION_PER_LOT
    
    if result == 1:
        wins += 1
        profit = (sl_pips * PIP_VALUE_USD * lot_size) - comm
    else:
        losses += 1
        profit = -(sl_pips * PIP_VALUE_USD * lot_size) - comm
        
    total_pnl_usd += profit
    year = row['signal_time'].year
    years_pnl[year] = years_pnl.get(year, 0) + profit

print(f"\nВсего сделок: {wins + losses}")
if (wins + losses) > 0:
    print(f"Винрейт: {(wins/(wins+losses))*100:.2f}%")
print(f"Чистый PnL: ${total_pnl_usd:.2f}")
print("\nРазбивка по годам:")
for y, pnl in sorted(years_pnl.items()):
    print(f"  {y}: ${pnl:.2f}")