import pytest
import pandas as pd
import duckdb
from pathlib import Path

## Проверка чтения сэмпла
def test_sample_data_loading():
    sample_path = Path("data/sample/EURUSD_sample.parquet")
    assert sample_path.exists(), "Sample data file is missing!"
    
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{sample_path}') LIMIT 10").fetchdf()
    
    assert not df.empty, "DataFrame is empty"
    assert 'bid' in df.columns, "Missing 'bid' column"
    assert 'ask' in df.columns, "Missing 'ask' column"

## Проверка логики расчета уровней WRN
def test_wrn_level_calculation():
    # Имитируем логику WRN из ноутбука
    price = 1.12345
    wrn_step = 0.0050
    expected_wrn = 1.12500 # Округление 1.12345 до ближайшего шага 0.0050
    
    # Наша функция округления
    calculated_wrn = round(price / wrn_step) * wrn_step
    
    assert pytest.approx(calculated_wrn, 0.0001) == expected_wrn, "WRN calculation logic failed"