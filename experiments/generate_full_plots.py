import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Настройки
SYMBOL = "EURUSD"
BASE_PATH = Path(f"data/parquet/{SYMBOL}/*.parquet")
PLOTS_PATH = Path("results/plots")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)
COMMISSION = 6.0
PIP_SIZE = 0.0001
PIP_VALUE = 10.0

print("Загрузка и агрегация полных данных (это займет время)...")
con = duckdb.connect()
query = f"""
WITH base_data AS (
    SELECT time_bucket(INTERVAL '15 minutes', timestamp) AS bar_time,
        arg_min(bid, timestamp) AS open, max(bid) AS high, min(bid) AS low, arg_max(bid, timestamp) AS close, 
        avg(ask - bid) AS avg_spread
    FROM read_parquet('{BASE_PATH}') GROUP BY bar_time
),
with_prev AS (
    SELECT *, 
        LAG(open) OVER (ORDER BY bar_time) AS open_1, LAG(close) OVER (ORDER BY bar_time) AS close_1,
        LAG(high) OVER (ORDER BY bar_time) AS high_1, LAG(low) OVER (ORDER BY bar_time) AS low_1
    FROM base_data
)
SELECT * FROM with_prev WHERE 
    (close_1 < open_1) AND (close > open) AND (close > open_1) -- Engulfing
    AND (low_1 <= low) AND (close > high_1) -- Sweep+Shift
ORDER BY bar_time;
"""
df = con.execute(query).fetchdf()
print(f"Найдено сетапов: {len(df)}")

# Симуляция
trades = []
for idx, row in df.iterrows():
    if idx + 50 >= len(df): break
    entry = row['close']
    sl = row['low_1'] - (2 * PIP_SIZE)
    sl_dist = entry - sl
    
    if sl_dist < 5 * PIP_SIZE or sl_dist > 60 * PIP_SIZE: continue
    tp = entry + sl_dist
    
    future = df.iloc[idx+1 : idx+50]
    result = None
    for _, f_row in future.iterrows():
        if f_row['low'] <= sl: result = 0; break
        if f_row['high'] >= tp: result = 1; break
        
    if result is not None:
        trades.append({
            'year': row['bar_time'].year,
            'result': result,
            'sl_pips': sl_dist / PIP_SIZE,
            'spread_pips': row['avg_spread'] / PIP_SIZE
        })

trades_df = pd.DataFrame(trades)
if len(trades_df) == 0:
    print("Нет сделок для графиков.")
    exit()

# 1. Winrate by Year
plt.figure(figsize=(8, 4))
winrate_by_year = trades_df.groupby('year')['result'].mean() * 100
sns.barplot(x=winrate_by_year.index, y=winrate_by_year.values)
plt.axhline(53.5, color='r', linestyle='--', label='Breakeven (with comm)')
plt.title(f"{SYMBOL} - Winrate by Year")
plt.ylabel("Winrate (%)")
plt.legend()
plt.savefig(PLOTS_PATH / "winrate_by_year.png")
plt.close()

# 2. PnL Curve
trades_df['pnl_pips'] = trades_df.apply(lambda r: r['sl_pips'] if r['result'] == 1 else -r['sl_pips'], axis=1)
trades_df['cumulative_pnl'] = trades_df['pnl_pips'].cumsum()
# Вычитаем комиссию (примерно 0.6 пипса за сделку на 1 лот)
trades_df['cumulative_pnl_net'] = trades_df['cumulative_pnl'] - (trades_df.index * 0.6)

plt.figure(figsize=(10, 4))
plt.plot(trades_df.index, trades_df['cumulative_pnl_net'], color='red')
plt.title(f"{SYMBOL} - Cumulative PnL (Net of Commissions)")
plt.xlabel("Trade Number")
plt.ylabel("PnL (Pips)")
plt.savefig(PLOTS_PATH / "pnl_curve.png")
plt.close()

# 3. Cost Distribution
plt.figure(figsize=(8, 4))
sns.histplot(trades_df['spread_pips'], bins=30, kde=True)
plt.title(f"{SYMBOL} - Spread Distribution at Entry")
plt.xlabel("Spread (Pips)")
plt.savefig(PLOTS_PATH / "cost_distribution.png")
plt.close()

print("Графики сохранены в results/plots/")