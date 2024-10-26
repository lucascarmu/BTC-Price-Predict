from app.utils import get_binance_klines
from datetime import datetime
import os


symbol = 'BTCUSDT'
interval = '1d'
start_date = '2017-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

df = get_binance_klines(symbol, interval, start_date, end_date)

if df is not None:
    # Define the full path to the file in the data folder
    data_dir = './data/'
    csv_file = os.path.join(data_dir, 'binance_btcusdt_price_history.csv')
    
    # Save the data to a CSV file in the data folder
    df.to_csv(csv_file, index=False)
    print(f"Historical Bitcoin price data from Binance downloaded and saved to '{csv_file}'")
    
