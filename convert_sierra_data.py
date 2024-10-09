import os
import struct
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone

from datetime import datetime, timezone

def read_scid_file(file_path):
    with open(file_path, 'rb') as f:
        header = f.read(144)
        version, symbol = struct.unpack('=i32s', header[:36])
        symbol = symbol.decode().strip('\x00')
        
        f.seek(144)
        
        data = []
        while True:
            record = f.read(40)
            if not record:
                break
            
            timestamp, open_price, high, low, close, volume, num_trades, bid_vol, ask_vol = struct.unpack('=qfffflllf', record)
            
            # Print the raw timestamp for debugging
            print(f"Raw timestamp: {timestamp}")
            
            # Assume the timestamp is in nanoseconds and divide by 1e9
            try:
                timestamp = timestamp / 1e9
                dt = datetime.fromtimestamp(timestamp, timezone.utc)
            except OSError:
                print(f"Invalid timestamp: {timestamp}. Skipping this record.")
                continue
            
            data.append({
                'timestamp': dt,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'num_trades': num_trades,
                'buy_volume': bid_vol,
                'sell_volume': ask_vol
            })
    
    return pd.DataFrame(data)


def read_depth_file(file_path):
    with open(file_path, 'rb') as f:
        header = f.read(56)
        version, symbol, reserved, num_records = struct.unpack('=i32s16si', header)
        
        data = []
        for _ in range(num_records):
            record = f.read(56)
            timestamp, price, size, side, position, is_end_of_refresh = struct.unpack('=qddifi', record)
            # Assume timestamp is in seconds since 1970-01-01 (UNIX epoch)
            dt = datetime.utcfromtimestamp(timestamp)
            
            data.append({
                'timestamp': dt,
                'price': price,
                'size': size,
                'side': 'bid' if side == 1 else 'ask',
                'position': position,
                'is_end_of_refresh': bool(is_end_of_refresh)
            })
    
    return pd.DataFrame(data)

def process_depth_data(df, levels=10):
    df = df.sort_values(['timestamp', 'side', 'price'])
    grouped = df.groupby(['timestamp', 'side'])
    bids = grouped.apply(lambda x: x[x['side'] == 'bid'].head(levels)).reset_index(drop=True)
    asks = grouped.apply(lambda x: x['side'] == 'ask').head(levels).reset_index(drop=True)
    
    bids_pivot = bids.pivot(index='timestamp', columns='position', values=['price', 'size'])
    asks_pivot = asks.pivot(index='timestamp', columns='position', values=['price', 'size'])
    
    bids_pivot.columns = [f'bid_price_{i}' if 'price' in col else f'bid_size_{i}' for i, col in enumerate(bids_pivot.columns, 1)]
    asks_pivot.columns = [f'ask_price_{i}' if 'price' in col else f'ask_size_{i}' for i, col in enumerate(asks_pivot.columns, 1)]
    
    return pd.concat([bids_pivot, asks_pivot], axis=1).reset_index()

def convert_to_parquet(input_file, output_file):
    file_extension = os.path.splitext(input_file)[1].lower()
    
    if file_extension == '.scid':
        df = read_scid_file(input_file)
    elif file_extension == '.depth':
        df = read_depth_file(input_file)
        df = process_depth_data(df)
    else:
        print(f"Unsupported file type: {file_extension}")
        return
    
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
    print(f"Converted {input_file} to {output_file}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

def convert_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.scid', '.depth')):
            input_file = os.path.join(directory, filename)
            output_file = os.path.join(directory, f"{os.path.splitext(filename)[0]}.parquet")
            convert_to_parquet(input_file, output_file)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.getcwd()
    
    print(f"Processing files in directory: {directory}")
    convert_files_in_directory(directory)
    print("Conversion complete!")