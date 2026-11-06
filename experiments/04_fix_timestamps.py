import duckdb
from pathlib import Path

con = duckdb.connect(':memory:')
base_dir = Path(__file__).resolve().parent / "data" / "parquet"

# Находим все файлы Parquet во всех папках
files = list(base_dir.glob("*/*.parquet"))

if not files:
    print("Файлы не найдены. Проверьте путь.")
    exit()

print(f"Найдено {len(files)} файлов для исправления...")

for file in files:
    print(f"Исправляем {file.name}...")
    temp_file = file.with_suffix('.tmp.parquet')
    
    # Читаем старый файл, делим время на 2 и сохраняем во временный файл
    con.execute(f"""
        COPY (
            SELECT 
                to_timestamp(epoch(timestamp) / 2.0) AS timestamp,
                bid, 
                ask, 
                volume
            FROM read_parquet('{file}')
        ) TO '{temp_file}' (FORMAT PARQUET, COMPRESSION SNAPPY);
    """)
    
    # Заменяем старый файл новым исправленным
    temp_file.replace(file)

print("Все файлы успешно исправлены! Теперь даты правильные.")
