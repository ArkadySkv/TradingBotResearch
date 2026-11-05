import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

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
TP_MULTIPLIER = 1.5 # Пробуем 1:1.5

print(f"=== ТЕСТ С ORDER FLOW (15M, Риск $50, TP 1:{TP_MULTIPLIER}) ===")

con = duckdb.connect(':memory:')

# SQL: Считаем Order Flow Imbalance (OFI) из тиков!
query_dataset = f"""
WITH raw_ticks AS (
    SELECT timestamp, bid, ask, volume,
        -- Считаем дельту (OFI) для каждого тика
        CASE 
            WHEN bid > LAG(bid) OVER (ORDER BY timestamp) THEN volume 
            WHEN bid < LAG(bid) OVER (ORDER BY timestamp) THEN -volume 
            ELSE 0 
        END AS tick_delta
    FROM read_parquet('{base_path}')
),
base_data AS (
    SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
        avg(ask - bid) AS avg_spread,
        SUM(tick_delta) AS bar_delta -- <--- НАШ ORDER FLOW
    FROM raw_ticks GROUP BY bar_time
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
        LAG(high) OVER (ORDER BY bar_time) AS high_1, LAG(low) OVER (ORDER BY bar_time) AS low_1,
        LAG(bar_delta) OVER (ORDER BY bar_time) AS delta_1
    FROM with_avg_atr
),
raw_signals AS (
    SELECT 
        p.bar_time AS signal_time, p.close AS entry_price, p.avg_spread, p.low, p.high, p.atr,
        p.bar_delta, p.delta_1,
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
)
SELECT DISTINCT ON (signal_time) * FROM raw_signals 
WHERE (raid_wrn_long = 1 OR raid_wrn_short = 1) AND (moe_bull_engulf=1 OR moe_bear_engulf=1)
ORDER BY signal_time;
"""
signals = con.execute(query_dataset).fetchdf()
bars_df = con.execute(f"SELECT time_bucket(INTERVAL '{TIMEFRAME} minutes', timestamp) AS bar_time, arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;").fetchdf()
bars_df.set_index('bar_time', inplace=True)

X_data, Y_data, details = [], [], []
pip_size = cfg['pip_size']

for idx, row in signals.iterrows():
    entry = row['entry_price']
    signal_idx = bars_df.index.get_loc(row['signal_time'])
    if signal_idx < 1: continue
    
    prev_bar = bars_df.iloc[signal_idx - 1]
    direction = 1 if row['sweep_shift_long'] == 1 else -1
    
    if direction == 1:
        sl = prev_bar['low'] - (2 * pip_size) 
        sl_dist = entry - sl
        tp = entry + (sl_dist * TP_MULTIPLIER)
    else:
        sl = prev_bar['high'] + (2 * pip_size)
        sl_dist = sl - entry
        tp = entry - (sl_dist * TP_MULTIPLIER)
        
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
        
    # ДОБАВЛЯЕМ ORDER FLOW В ПРИЗНАКИ
    X_data.append([
        row['moe_bull_engulf'], row['moe_bear_engulf'], 
        row['sweep_shift_long'], row['sweep_shift_short'],
        row['avg_spread'] / pip_size, sl_pips, row['atr'] / pip_size,
        row['bar_delta'], row['delta_1'] # <--- МАГИЯ ЗДЕСЬ
    ])
    Y_data.append(result)
    details.append({"time": row['signal_time'], "year": row['signal_time'].year, "sl_pips": sl_pips, "result": result})

X_real = np.array(X_data)
Y_real = np.array(Y_data)
details_df = pd.DataFrame(details)

test_years = [2022, 2023, 2024, 2025]
overall_trades, overall_wins, overall_pnl_usd = 0, 0, 0

for test_year in test_years:
    train_mask = details_df['year'] < test_year
    test_mask = details_df['year'] == test_year
    
    if sum(train_mask) < 500 or sum(test_mask) < 50: continue
        
    X_train, Y_train = X_real[train_mask], Y_real[train_mask]
    X_test = X_real[test_mask]
    details_test = details_df[test_mask].reset_index(drop=True)
    
    clf = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
    clf.fit(X_train, Y_train)
    
    y_prob = clf.predict_proba(X_test)[:, 1]
    selected_trades = details_test[y_prob >= 0.55] # Оставляем высокую уверенность
    
    if len(selected_trades) == 0: continue
        
    wins = (selected_trades['result'] == 1).sum()
    total_pnl_usd = 0
    
    for _, trade in selected_trades.iterrows():
        lot_size = FIXED_RISK_USD / (trade['sl_pips'] * PIP_VALUE_USD)
        lot_size = max(round(lot_size, 2), 0.01)
        comm = lot_size * COMMISSION_PER_LOT
        
        if trade['result'] == 1:
            profit = (trade['sl_pips'] * TP_MULTIPLIER * PIP_VALUE_USD * lot_size) - comm
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