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
time_offset_sql = "INTERVAL '1 hour'" 
COMMISSION_PER_LOT = 6.0 

TIMEFRAME = 30

print(f"=== ФИНАЛЬНЫЙ БЭКТЕСТ (Только лучшие сетапы + Breakeven) ===")

con = duckdb.connect(':memory:')

# SQL: Строгий фильтр (Только Doji и Engulfing в NY/London/Asia)
query_dataset = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time,
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
    CROSS JOIN UNNEST(generate_series(
        CAST(floor(w.prev_low / {cfg['wrn_step']}) AS BIGINT) - 1, 
        CAST(ceil(w.prev_high / {cfg['wrn_step']}) AS BIGINT) + 1, 
        1
    )) AS t(level_int)
),
session_opens AS (
    SELECT date_trunc('day', bar_time + {time_offset_sql}) AS adj_day, arg_min(open, bar_time) AS do_level
    FROM base_data WHERE EXTRACT(hour FROM (bar_time + {time_offset_sql})) = 16 GROUP BY adj_day
),
with_prev AS (
    SELECT *, 
        LAG(open) OVER (ORDER BY bar_time) AS open_1, 
        LAG(close) OVER (ORDER BY bar_time) AS close_1,
        LAG(high) OVER (ORDER BY bar_time) AS high_1,
        LAG(low) OVER (ORDER BY bar_time) AS low_1
    FROM base_data
),
raw_signals AS (
    SELECT 
        p.bar_time AS signal_time, p.close AS entry_price, p.avg_spread, p.low, p.high,
        
        -- MoE
        CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS moe_engulf,
        CASE WHEN ABS(p.close - p.open) <= (p.high - p.low) * 0.25 AND (p.high - p.low) > 0 THEN 1 ELSE 0 END AS moe_doji,
        
        -- RAID
        CASE WHEN p.low_1 <= w.wrn_level AND p.close > w.wrn_level THEN 1 ELSE 0 END AS raid_wrn,
        CASE WHEN p.low_1 <= s.do_level AND p.close > s.do_level THEN 1 ELSE 0 END AS raid_do,
        
        -- SWEEP + SHIFT
        CASE WHEN p.low_1 <= p.low AND p.close > p.high_1 THEN 1 ELSE 0 END AS sweep_shift_long,
        
        -- Sessions (1=Asia, 2=Frankfurt, 3=London, 4=NY)
        CASE 
            WHEN EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= 3 AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < 11 THEN 1
            WHEN EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= 11 AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < 12 THEN 2
            WHEN EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= 12 AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < 16 THEN 3
            ELSE 4 
        END AS session
    FROM with_prev p
    LEFT JOIN wrn_levels w ON p.bar_time = w.bar_time
    LEFT JOIN session_opens s ON date_trunc('day', p.bar_time + {time_offset_sql}) = s.adj_day
)
SELECT DISTINCT ON (signal_time) * FROM raw_signals 
WHERE (raid_wrn = 1 OR raid_do = 1) 
  AND (moe_engulf=1 OR moe_doji=1)
  -- ФИЛЬТР: Только лучшие сессии (Азия, Лондон, NY). Франкфурт исключаем.
  AND session IN (1, 3, 4) 
ORDER BY signal_time;
"""
signals = con.execute(query_dataset).fetchdf()

bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# Финансовый расчет с БЕЗУБЫТКОМ (Breakeven)
wins, losses, breakevens = 0, 0, 0
total_pnl_pips = 0
pip_size = cfg['pip_size']

print("Симуляция с Breakeven (перенос стопа в 0 после +10 пипсов)...")

for idx, row in signals.iterrows():
    entry = row['entry_price']
    signal_idx = bars_df.index.get_loc(row['signal_time'])
    if signal_idx < 1: continue
    
    prev_bar = bars_df.iloc[signal_idx - 1]
    sl = prev_bar['low'] - (2 * pip_size) 
    sl_dist = entry - sl
    
    if sl_dist < 5 * pip_size or sl_dist > 60 * pip_size: continue
    
    # TP = 1.2 (чуть больше стопа)
    tp = entry + (sl_dist * 1.2)
    
    # Логика Breakeven: как только цена проходит 1 размер стопа вверх, ставим SL в 0
    be_trigger_price = entry + sl_dist
    be_moved = False
    
    future_bars = bars_df.iloc[signal_idx + 1: signal_idx + 100]
    result = None
    
    for t, b in future_bars.iterrows():
        # Проверяем, не выбило ли стоп
        if b['low'] <= sl:
            if be_moved:
                result = "BE" # Безубыток (0 пипсов)
            else:
                result = "LOSS"
            break
        # Проверяем, не достигли ли Тейка
        if b['high'] >= tp: 
            result = "WIN"
            break
        # Если цена дошла до уровня BE, переносим стоп
        if b['high'] >= be_trigger_price:
            be_moved = True
            sl = entry # Переносим в 0
            
    if result is None: continue
        
    sl_pips = (entry - (prev_bar['low'] - 2 * pip_size)) / pip_size # Оригинальный размер стопа
    
    if result == "WIN":
        wins += 1
        total_pnl_pips += (sl_pips * 1.2)
    elif result == "LOSS":
        losses += 1
        total_pnl_pips -= sl_pips
    elif result == "BE":
        breakevens += 1
        # 0 пипсов

total_trades = wins + losses + breakevens
if total_trades == 0:
    print("Нет сделок.")
    exit()

win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
gross_pnl_usd = total_pnl_pips * 10.0
commission = total_trades * COMMISSION_PER_LOT
net_pnl_usd = gross_pnl_usd - commission

print(f"\n--- РЕЗУЛЬТАТ ---")
print(f"Всего сделок: {total_trades}")
print(f"Выигрышей: {wins} | Проигрышей: {losses} | Безубытков: {breakevens}")
print(f"Винрейт (без BE): {win_rate:.2f}%")
print(f"Валовая прибыль: ${gross_pnl_usd:.2f}")
print(f"Комиссия (Vantage): ${commission:.2f}")
print(f"ЧИСТАЯ ПРИБЫЛЬ: ${net_pnl_usd:.2f}")