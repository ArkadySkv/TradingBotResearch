import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================================
# НАСТРОЙКИ
# =====================================================================
SYMBOLS = ["EURUSD", "USDJPY", "GER40"]
TIMEFRAMES = [15, 30, 60]
CONFIG = {
    "EURUSD": {"pip_size": 0.0001, "wrn_step": 0.0010},
    "USDJPY": {"pip_size": 0.01, "wrn_step": 0.50},
    "GER40": {"pip_size": 1.0, "wrn_step": 500.0}
}
COMMISSION_PER_LOT = 6.0
FIXED_RISK_USD = 50.0
PIP_VALUE_USD = 10.0  # для простоты

def run_test(symbol, tf):
    cfg = CONFIG[symbol]
    base_path = Path(__file__).resolve().parent / "data" / "parquet" / symbol / "*.parquet"
    time_offset_sql = "INTERVAL '1 hour'" 
    
    con = duckdb.connect(':memory:')
    
    # SQL без ML, просто Sweep + Engulfing
    query = f"""
    WITH base_data AS (
        SELECT time_bucket(INTERVAL '{tf} minutes', timestamp) AS bar_time,
            arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
            avg(ask - bid) AS avg_spread
        FROM read_parquet('{base_path}') GROUP BY bar_time
    ),
    with_prev AS (
        SELECT *, 
            LAG(open) OVER (ORDER BY bar_time) AS open_1, LAG(close) OVER (ORDER BY bar_time) AS close_1,
            LAG(high) OVER (ORDER BY bar_time) AS high_1, LAG(low) OVER (ORDER BY bar_time) AS low_1
        FROM base_data
    )
    SELECT 
        bar_time, close, low, high, avg_spread,
        CASE WHEN (close_1 < open_1) AND (close > open) AND (close > open_1) THEN 1 ELSE 0 END AS bull_engulf,
        CASE WHEN (high_1 >= high) AND (close < low_1) THEN 1 ELSE 0 END AS sweep_shift_short,
        CASE WHEN (low_1 <= low) AND (close > high_1) THEN 1 ELSE 0 END AS sweep_shift_long
    FROM with_prev
    WHERE (bull_engulf = 1 OR sweep_shift_long = 1 OR sweep_shift_short = 1)
    ORDER BY bar_time;
    """
    signals = con.execute(query).fetchdf()
    bars_df = con.execute(f"SELECT time_bucket(INTERVAL '{tf} minutes', timestamp) AS bar_time, arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close FROM read_parquet('{base_path}') GROUP BY bar_time ORDER BY bar_time;").fetchdf()
    bars_df.set_index('bar_time', inplace=True)
    
    wins, losses = 0, 0
    total_pnl_usd = 0
    pip_size = cfg['pip_size']
    
    for idx, row in signals.iterrows():
        entry = row['close']
        signal_idx = bars_df.index.get_loc(row['bar_time'])
        if signal_idx < 1: continue
        
        prev_bar = bars_df.iloc[signal_idx - 1]
        direction = 1 if row['sweep_shift_long'] == 1 else -1
        
        if direction == 1:
            sl = prev_bar['low'] - (2 * pip_size)
            sl_dist = entry - sl
            tp = entry + sl_dist
        else:
            sl = prev_bar['high'] + (2 * pip_size)
            sl_dist = sl - entry
            tp = entry - sl_dist
            
        sl_pips = sl_dist / pip_size
        if sl_pips < 3 or sl_pips > 100: continue
        
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
    
    total = wins + losses
    if total == 0:
        return None
    winrate = wins / total * 100
    return {"Symbol": symbol, "TF": f"{tf}M", "Trades": total, "Winrate": round(winrate, 1), "PnL": round(total_pnl_usd, 1)}

# Запускаем
results = []
for sym in SYMBOLS:
    for tf in TIMEFRAMES:
        print(f"Testing {sym} {tf}M...")
        res = run_test(sym, tf)
        if res: results.append(res)

df = pd.DataFrame(results)
print("\n--- РЕЗУЛЬТАТЫ БАЗОВОГО ТЕСТА (БЕЗ ML) ---")
print(df.to_string(index=False))