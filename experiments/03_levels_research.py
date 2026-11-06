import duckdb
import pandas as pd
from pathlib import Path

# =====================================================================
# НАСТРОЙКИ ИССЛЕДОВАНИЯ
# =====================================================================
SYMBOL = "EURUSD" 

# ВАЖНО: Мы задаем сессии в GMT+4 (как на ваших графиках TradingView)
# Если ваш брокер отдает время в GMT+3, скрипт сдвинет его на 1 час вперед.
CONFIG = {
    "EURUSD": {
        "wrn_step": 0.0010,
        "pip_size": 0.0001,
        # Сессии в GMT+4 (начало, конец)
        "asian": (3, 11),
        "london": (12, 16),
        "ny": (16, 24), # Ночь с 16 до 00
        "preferred_session": "ny" # В какой сессии мы хотим торговать
    },
    "USDJPY": {
        "wrn_step": 1.0,
        "pip_size": 0.01,
        "asian": (3, 11),
        "london": (12, 16),
        "ny": (16, 24),
        "preferred_session": "asian"
    },
    "GER40": {
        "wrn_step": 500.0,
        "pip_size": 1.0,
        "asian": (5, 11),
        "frankfurt": (11, 12),
        "london": (12, 16),
        "ny": (16, 24),
        "preferred_session": "london"
    }
}

cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
con = duckdb.connect(':memory:')

print(f"=== Исследование {SYMBOL} | Предпочтительная сессия: {cfg['preferred_session'].upper()} ===")

# Сдвиг времени: Если MT5 отдает GMT+3, а нам нужно GMT+4, добавляем 1 час.
time_offset_sql = "INTERVAL '1 hour'" 

# Определяем границы сессии для фильтрации
start_hour, end_hour = cfg[cfg["preferred_session"]]

query = f"""
WITH base_data AS (
    SELECT 
        time_bucket(INTERVAL '5 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open,
        max(bid) AS high,
        min(bid) AS low,
        arg_max(bid, timestamp) AS close,
        avg(ask - bid) AS avg_spread
    FROM read_parquet('{base_path}')
    GROUP BY bar_time
),
-- 1. Расчет WRN (как раньше)
weekly_range AS (
    SELECT date_trunc('week', bar_time) AS week_start, max(high) AS prev_high, min(low) AS prev_low
    FROM base_data GROUP BY week_start
),
wrn_levels AS (
    SELECT 
        b.bar_time,
        CAST(t.level_int AS DOUBLE) * {cfg['wrn_step']} AS wrn_level
    FROM base_data b
    JOIN weekly_range w ON date_trunc('week', b.bar_time) = w.week_start + INTERVAL '7 days'
    CROSS JOIN UNNEST(generate_series(
        CAST(floor(w.prev_low / {cfg['wrn_step']}) AS BIGINT), 
        CAST(ceil(w.prev_high / {cfg['wrn_step']}) AS BIGINT), 1
    )) AS t(level_int)
),
-- 2. Расчет DO (Открытие выбранной сессии)
session_opens AS (
    SELECT 
        date_trunc('day', bar_time + {time_offset_sql}) AS adj_day,
        arg_min(open, bar_time) AS do_level
    FROM base_data
    WHERE EXTRACT(hour FROM (bar_time + {time_offset_sql})) = {start_hour}
    GROUP BY adj_day
),
-- 3. Поиск Morning Star
with_prev AS (
    SELECT *,
        LAG(open) OVER (ORDER BY bar_time) AS open_1,
        LAG(close) OVER (ORDER BY bar_time) AS close_1,
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
-- 4. Финальный отбор: Ищем только внутри выбранной сессии!
SELECT 
    p.bar_time AS entry_time,
    p.close AS entry_price,
    p.avg_spread,
    'WRN' AS level_type,
    w.wrn_level AS level_price
FROM pattern_ms p
JOIN wrn_levels w ON p.bar_time = w.bar_time
WHERE p.is_ms = TRUE 
  AND p.low_2 <= w.wrn_level 
  AND p.close > w.wrn_level
  -- ФИЛЬТР СЕССИИ: Берем только часы внутри выбранного окна
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) >= {start_hour}
  AND EXTRACT(hour FROM (p.bar_time + {time_offset_sql})) < {end_hour}
ORDER BY entry_time
LIMIT 20;
"""

try:
    signals_df = con.execute(query).fetchdf()
    if len(signals_df) > 0:
        print(f"\nНайдено {len(signals_df)} сигналов (WRN) в сессии {cfg['preferred_session'].upper()}:")
        print(signals_df.to_string(index=False))
    else:
        print("\n[ВНИМАНИЕ] Сигналов не найдено.")
except Exception as e:
    print(f"Ошибка SQL: {e}")
