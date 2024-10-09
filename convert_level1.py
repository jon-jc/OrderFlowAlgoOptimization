import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def convert_level1_to_parquet(csv_file, parquet_file):
    # Read the CSV file
    df = pd.read_csv(csv_file, parse_dates=['DateTime'])
    
    # Rename columns to match our strategy requirements
    df = df.rename(columns={
        'DateTime': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # If buy_volume and sell_volume are not available, you can estimate them
    # This is a simple estimation and may not be accurate
    if 'buy_volume' not in df.columns:
        df['buy_volume'] = df['volume'] * (df['close'] > df['open']).astype(int)
        df['sell_volume'] = df['volume'] - df['buy_volume']
    
    # Ensure all required columns are present
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the data")
    
    # Convert to Parquet
    table = pa.Table.from_pandas(df[required_columns])
    pq.write_table(table, parquet_file)

# Usage
convert_level1_to_parquet('path_to_your_level1_csv.csv', 'level1_data.parquet')