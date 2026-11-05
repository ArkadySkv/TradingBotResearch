from mt5_remote import MetaTrader5

# Initialize the remote connection (defaults to localhost:18812)
mt5 = MetaTrader5(host='192.168.122.139', port=18812)
if not mt5.initialize():
    print("Initialization failed. Check if the Windows server is running.")
else:
    print("Successfully connected to MT5 from Lubuntu!")
    # Test fetching data
    account_info = mt5.account_info()
    print(f"Account Balance: {account_info.balance}")
    
    # Fetch a few ticks for EURUSD
    ticks = mt5.copy_ticks_from("EURUSD", 0, 10) 
    print(f"Fetched {len(ticks)} ticks.")
    mt5.shutdown()