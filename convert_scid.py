
import struct
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta

def read_scid_file(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        header = f.read(144)
        version, symbol = struct.unpack('=i32s', header[:36])
        symbol = symbol.decode().strip('\x00')
        
        # Move to the start of the data
        f.seek(144)
        
        data = []
        while True:
            record = f.read(40)
            if not record:
                break
            
            timestamp, open_price, high, low, close, volume, num_trades, bid_vol, ask_vol = struct.unpack('=qfffflllf', record)
            
            # Convert timestamp to datetime
            dt = datetime(1899, 12, 30) + timedelta(days=timestamp)
            
            data.append({
                'timestamp': dt,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'num_trades': num_trades,
                'bid_volume': bid_vol,
                'ask_volume': ask_vol
            })
    
    return pd.DataFrame(data)

def convert_scid_to_parquet(scid_file, parquet_file):
    # Read .SCID file
    df = read_scid_file(scid_file)
    
    # Rename columns to match our strategy requirements
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'bid_volume': 'buy_volume',
        'ask_volume': 'sell_volume'
    })
    
    # Ensure all required columns are present
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the data")
    
    # Convert to Parquet
    table = pa.Table.from_pandas(df[required_columns])
    pq.write_table(table, parquet_file)

    print(f"Converted {scid_file} to {parquet_file}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

# Usage
scid_file = 'path_to_your_file.scid'
parquet_file = 'level1_data.parquet'
convert_scid_to_parquet(scid_file, parquet_file)

# Verify the converted data
def verify_parquet(parquet_file):
    table = pq.read_table(parquet_file)
    print("Columns:", table.column_names)
    print("Number of rows:", table.num_rows)
    print("First few rows:")
    print(table.to_pandas().head())

verify_parquet(parquet_file)
