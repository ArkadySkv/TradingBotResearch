from mt5_remote import Meta Trader5 as mt5
import duckdb
import pandas as pd
from datetime import datetime
from pathlib import Path
import time

### WINDOWS SCRIPT

class ProjectPaths:
    """Centralizes path management to prevent 'spaghetti' file paths."""
    def __init__(self):
        # Resolves to the trading_project root folder
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent 
        self.DATA_DIR = self.BASE_DIR / "data"
        self.PARQUET_DIR = self.DATA_DIR / "parquet"
        
        # Ensure directories exist
        self.PARQUET_DIR.mkdir(parents=True, exist_ok=True)

PATHS = ProjectPaths()

SYMBOLS = ["EURUSD", "USDJPY", "GER40"] 
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2026, 1, 1)

def download_and_process_ticks(symbol: str, start: datetime, end: datetime, db_conn: duckdb.DuckDBPyConnection):
    print(f"\n[FETCHING] {symbol} from {start.date()} to {end.date()} via MT5 Bridge...")
    
    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Failed to select {symbol} in MT5.")
        return

    # 1. Fetch raw ticks from Windows VM
    ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
    
    if ticks is None or len(ticks) == 0:
        print(f"[ERROR] No ticks received for {symbol}: {mt5.last_error()}")
        return
        
    # We use Pandas purely as a in-memory bridge (zero-copy to DuckDB)
    df = pd.DataFrame(ticks)
    
    # 2. Register the DataFrame into DuckDB's memory
    db_conn.register('raw_ticks', df)

    # 3. Define output path and execute the DuckDB COPY command
    symbol_dir = PATHS.PARQUET_DIR / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    # Partition file by year range
    file_name = f"{symbol}_{start.year}_{end.year}.parquet"
    output_path = symbol_dir / file_name
    
    print(f"[DUCKDB] Writing to {output_path}...")
    
    # Convert epoch time to native SQL TIMESTAMP within DuckDB
    query = f"""
        COPY (
            SELECT 
                to_timestamp(time + time_msc / 1000.0) AS timestamp,
                bid,
                ask,
                volume
            FROM raw_ticks
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION SNAPPY);
    """
    
    try:
        db_conn.execute(query)
        
        # Verify row count directly from the newly written Parquet file
        row_count = db_conn.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
        print(f"[SUCCESS] Saved {row_count:,} ticks to {file_name}")
        
    except Exception as e:
        print(f"[ERROR] DuckDB failed to write Parquet: {e}")
    finally:
        # Unregister the table to free up memory
        db_conn.unregister('raw_ticks')


def main():
    # Connect to MT5 via the Windows VM bridge
    if not mt5.initialize():
        print(f"[ERROR] MT5 bridge initialization failed. Is the Windows VM server running?")
        return

    account_info = mt5.account_info()
    if account_info:
        print(f"[INFO] Connected to MT5 Bridge. Account: {account_info.login}")

    # Initialize DuckDB (In-Memory connection for processing)
    con = duckdb.connect(database=':memory:')
    
    start_time = time.time()
    
    for symbol in SYMBOLS:
        download_and_process_ticks(symbol, START_DATE, END_DATE, con)
        
    con.close()
    mt5.shutdown()
    
    elapsed = time.time() - start_time
    print(f"\n[DONE] Data extraction and Parquet generation complete in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
