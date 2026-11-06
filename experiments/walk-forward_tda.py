import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape
from gtda.pipeline import make_pipeline

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
SYMBOL = "EURUSD" 
CONFIG = {"EURUSD": {"pip_size": 0.0001, "wrn_step": 0.0010}}
cfg = CONFIG[SYMBOL]
base_path = Path(__file__).resolve().parent / "data" / "parquet" / SYMBOL / "*.parquet"
time_offset_sql = "INTERVAL '1 hour'" 
COMMISSION_PER_LOT = 6.0 
TIMEFRAME = 15
FIXED_RISK_USD = 50.0 
PIP_VALUE_USD = 10.0  

print(f"=== TDA RESEARCH (15M, Sweep+Shift) ===")

con = duckdb.connect(':memory:')

# SQL: Находим сетапы и вытаскиваем последние 20 свечей (для TDA)
query_dataset = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
        avg(ask - bid) AS avg_spread, sum(volume) AS volume
    FROM read_parquet('{base_path}') GROUP BY bar_time
),
with_atr AS (
    SELECT *, GREATEST(high - low, ABS(high - LAG(close) OVER (ORDER BY bar_time)), ABS(low - LAG(close) OVER (ORDER BY bar_time))) AS tr
    FROM base_data
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
        p.bar_time AS signal_time, p.close AS entry_price, p.avg_spread, p.low, p.high, p.atr,
        CASE WHEN (p.close_1 < p.open_1) AND (p.close > p.open) AND (p.close > p.open_1) THEN 1 ELSE 0 END AS moe_bull_engulf,
        CASE WHEN (p.close_1 > p.open_1) AND (p.close < p.open) AND (p.close < p.open_1) THEN 1 ELSE 0 END AS moe_bear_engulf,
        CASE WHEN p.low_1 <= w.wrn_level AND p.close > w.wrn_level THEN 1 ELSE 0 END AS raid_wrn_long,
        CASE WHEN p.high_1 >= w.wrn_level AND p.close < w.wrn_level THEN 1 ELSE 0 END AS raid_wrn_short,
        CASE WHEN p.low_1 <= p.low AND p.close > p.high_1 THEN 1 ELSE 0 END AS sweep_shift_long,
        CASE WHEN p.high_1 >= p.high AND p.close < p.low_1 THEN 1 ELSE 0 END AS sweep_shift_short
    FROM with_prev p
    LEFT JOIN wrn_levels w ON p.bar_time = w.bar_time
)
SELECT DISTINCT ON (signal_time) * FROM raw_signals 
WHERE (raid_wrn_long = 1 OR raid_wrn_short = 1) AND (moe_bull_engulf=1 OR moe_bear_engulf=1)
ORDER BY signal_time;
"""
signals = con.execute(query_dataset).fetchdf()

# Загружаем полные данные для расчета TDA
bars_df = con.execute(f"""
    SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time, 
           arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close,
           sum(volume) as volume
    FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;
""").fetchdf()
bars_df.set_index('bar_time', inplace=True)

# Настройка TDA пайплайна
persistence = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=-1)
landscape = PersistenceLandscape(n_bins=10, n_layers=2)
tda_pipeline = make_pipeline(persistence, landscape)

X_basic, X_tda, Y_data, details = [], [], [], []
pip_size = cfg['pip_size']
WINDOW_SIZE = 20 # Берем 20 свечей до входа

print("Формирование данных и расчет TDA признаков (это может занять время)...")

for idx, row in signals.iterrows():
    entry = row['entry_price']
    signal_idx = bars_df.index.get_loc(row['signal_time'])
    if signal_idx < WINDOW_SIZE + 1: continue
    
    prev_bar = bars_df.iloc[signal_idx - 1]
    direction = 1 if row['sweep_shift_long'] == 1 else -1
    
    # Стоп и Тейк (1:1)
    if direction == 1:
        sl = prev_bar['low'] - (2 * pip_size) 
        sl_dist = entry - sl
        tp = entry + sl_dist
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

    # --- РАСЧЕТ TDA ---
    # Берем окно из 20 свечей перед сигналом
    window = bars_df.iloc[signal_idx - WINDOW_SIZE : signal_idx]
    
    # Создаем облако точек: (Индекс времени, Нормализованная цена, Нормализованный объем)
    # Нормализуем, чтобы TDA не ломался от масштаба
    point_cloud = np.column_stack((
        np.arange(len(window)),
        (window['close'].values - entry) / pip_size,
        (window['volume'].values - window['volume'].mean()) / (window['volume'].std() + 1e-5)
    ))
    
    # Ограничиваем размер, чтобы не переполнить память
    if len(point_cloud) > 100: point_cloud = point_cloud[-100:]
    
    X_tda.append(point_cloud)
    X_basic.append([
        row['moe_bull_engulf'], row['moe_bear_engulf'], 
        row['sweep_shift_long'], row['sweep_shift_short'],
        row['avg_spread'] / pip_size, sl_pips, row['atr'] / pip_size
    ])
    Y_data.append(result)
    details.append({"time": row['signal_time'], "year": row['signal_time'].year, "sl_pips": sl_pips, "result": result})

if len(X_tda) == 0:
    print("Недостаточно данных для TDA.")
    exit()

print(f"Найдено сетапов: {len(X_tda)}. Запускаем вычисление персистентной гомологии...")

# Применяем TDA
X_tda_raw = np.array(X_tda)
X_tda_features = tda_pipeline.fit_transform(X_tda_raw)
X_tda_flat = X_tda_features.reshape(X_tda_features.shape[0], -1)

# Объединяем базовые признаки и TDA
X_basic = np.array(X_basic)
X_combined = np.hstack((X_basic, X_tda_flat))
Y_real = np.array(Y_data)
details_df = pd.DataFrame(details)

print(f"Размерность TDA-признаков: {X_tda_flat.shape}")

# =====================================================================
# WALK-FORWARD С TDA
# =====================================================================
test_years = [2022, 2023, 2024, 2025]
overall_trades, overall_wins, overall_pnl_usd = 0, 0, 0

print("\n--- РЕЗУЛЬТАТЫ WALK-FORWARD С TDA ---")

for test_year in test_years:
    train_mask = details_df['year'] < test_year
    test_mask = details_df['year'] == test_year
    
    if sum(train_mask) < 300 or sum(test_mask) < 30: 
        print(f"[{test_year}] Мало данных. Пропуск.")
        continue
        
    X_train, Y_train = X_combined[train_mask], Y_real[train_mask]
    X_test = X_combined[test_mask]
    details_test = details_df[test_mask].reset_index(drop=True)
    
    clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
    clf.fit(X_train, Y_train)
    
    y_prob = clf.predict_proba(X_test)[:, 1]
    # Берем только те сделки, где модель уверена
    selected_trades = details_test[y_prob >= 0.54] 
    
    if len(selected_trades) == 0: continue
        
    wins = (selected_trades['result'] == 1).sum()
    total_pnl_usd = 0
    
    for _, trade in selected_trades.iterrows():
        lot_size = FIXED_RISK_USD / (trade['sl_pips'] * PIP_VALUE_USD)
        lot_size = max(round(lot_size, 2), 0.01)
        comm = lot_size * COMMISSION_PER_LOT
        
        if trade['result'] == 1:
            profit = (trade['sl_pips'] * PIP_VALUE_USD * lot_size) - comm
        else:
            profit = -(trade['sl_pips'] * PIP_VALUE_USD * lot_size) - comm
        total_pnl_usd += profit
        
    win_rate = (wins / len(selected_trades)) * 100
    print(f"[{test_year}] Сделок: {len(selected_trades)} | Винрейт: {win_rate:.2f}% | PnL: ${total_pnl_usd:.2f}")
    
    overall_trades += len(selected_trades)
    overall_wins += wins
    overall_pnl_usd += total_pnl_usd

print("-" * 40)
if overall_trades > 0:
    print(f"ИТОГО -> Сделок: {overall_trades} | Винрейт: {(overall_wins/overall_trades)*100:.2f}% | PnL: ${overall_pnl_usd:.2f}")