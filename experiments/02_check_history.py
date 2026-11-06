import MetaTrader5 as mt5
from datetime import datetime

mt5.initialize()

symbols = ["EURUSD", "USDJPY", "GER40", "HK50", "Nikkei225"]
years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

print("=== Диагностика доступной тиковой истории ===")
for symbol in symbols:
    # Активируем символ
    mt5.symbol_select(symbol, True)
    print(f"\n--- {symbol} ---")
    
    # Проверяем информацию о символе
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"  [ERROR] Символ {symbol} не найден.")
        continue
        
    print(f"  Видимый в Market Watch: {info.visible}")
    print(f"  Точек: {info.point}")
    
    for year in years:
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59)
        
        # Пробуем получить тики
        ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
        
        if ticks is None:
            print(f"  {year}: Ошибка -> {mt5.last_error()}")
        elif len(ticks) == 0:
            print(f"  {year}: 0 тиков (данных нет в кэше/на сервере)")
        else:
            print(f"  {year}: {len(ticks):,} тиков")

mt5.shutdown()
print("\n[INFO] Диагностика завершена.")