import duckdb
import pandas as pd
from pathlib import Path

# =====================================================================
# НАСТРОЙКИ ИССЛЕДОВАНИЯ
# =====================================================================
SYMBOL = "EURUSD" 
SL_PIPS = 20       # Стоп-лосс
TP_PIPS = 25       # Тейк-профит (1:1.25)
COMMISSION_PER_LOT = 6.0 # Комиссия Vantage RAW ($6 за круг)

CONFIG = {
    "EURUSD": {"wrn_step": 0.0010, "pip_size": 0.0001, "ny": (16, 24)},
    "USDJPY": {"wrn_step": 1.0, "pip_size": 0.01, "asian": (3, 11)},
    "GER40": {"wrn_step": 500.0, "pip_size": 1.0, "london": (12, 16)}
}

cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')

print(f"=== Бэктест {SYMBOL} | SL: {SL_PIPS} pips | TP: {TP_PIPS} pips ===")

time_offset_sql = "INTERVAL '1 hour'" 
start_hour, end_hour = cfg["ny"] # Меняем в зависимости от актива

# Шаг 1: Получаем сигналы (как в прошлом скрипте)
query_signals = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, avg(ask - bid) AS avg_spread
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
    SELECT *, LAG(open) OVER (ORDER BY bar_time) AS open_1, LAG(close) OVER (ORDER BY bar_time) AS close_1,
        LAG(open, 2) OVER (ORDER BY bar_time) AS open_2, LAG(close, 2) OVER (ORDER BY bar_time) AS close_2, LAG(low, 2) OVER (ORDER BY bar_time) AS low_2
    FROM base_data
),
pattern_ms AS (
    SELECT *, (close_2 < open_2) AND (ABS(close_1 - open_1) < (open_2 - close_2) * 0.3) AND (close > open) AND (close > (open_2 + close_2) / 2) AS is_ms
    FROM with_prev
)
SELECT p.bar_time AS entry_time, p.close AS entry_price, p.avg_spread, w.wrn_level AS level_price
FROM pattern_ms p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.is_ms = TRUE AND p.low_2 <= w.wrn_level AND p.close > w.wrn_level
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= {start_hour}
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < {end_hour}
ORDER BY entry_time;
"""
signals_df = con.execute(query_signals).fetchdf()
print(f"Найдено сигналов для симуляции: {len(signals_df):,}")

if len(signals_df) == 0:
    print("Нет сигналов. Завершаем.")
    exit()

# Шаг 2: Загружаем ВСЕ 5M свечи для симуляции будущего движения
print("Загрузка 5M свечей для симуляции...")
bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# Шаг 3: Симуляция
pip_size = cfg['pip_size']
sl_dist = SL_PIPS * pip_size
tp_dist = TP_PIPS * pip_size

wins, losses, unresolved = 0, 0, 0
total_pnl_pips = 0

print("Симуляция сделок...")
for idx, row in signals_df.iterrows():
    entry_time = row['entry_time']
    # Вход по Ask (добавляем спред к цене Bid, так как мы покупаем)
    entry_price = row['entry_price'] + row['avg_spread'] 
    
    tp_price = entry_price + tp_dist
    sl_price = entry_price - sl_dist
    
    # Смотрим на 100 свечей (500 минут) вперед
    future_bars = bars_df.loc[entry_time:].iloc[1:100]
    
    trade_result = None
    for future_time, future_row in future_bars.iterrows():
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
# Для EURUSD 1 пипс на лот 1.0 = $10
gross_pnl_usd = total_pnl_pips * 10.0
commission = total_trades * COMMISSION_PER_LOT
net_pnl_usd = gross_pnl_usd - commission

print(f"\n--- РЕЗУЛЬТАТЫ БЭКТЕСТА ---")
print(f"Закрытых сделок: {total_trades}")
print(f"Не закрытых (зависших): {unresolved}")
print(f"Выигрышей (WIN): {wins}")
print(f"Проигрышей (LOSS): {losses}")
print(f"Винрейт: {win_rate:.2f}%")
print(f"Чистый PnL (с учетом комиссии): ${net_pnl_usd:.2f}")

if net_pnl_usd > 0:
    print("\n[УСПЕХ] Стратегия показывает положительное матожидание на этих данных!")
else:
    print("\n[ПРОВАЛ] Стратегия убыточна при данных параметрах. Нужны фильтры.")
