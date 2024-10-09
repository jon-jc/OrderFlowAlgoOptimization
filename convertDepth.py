import struct
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta

def read_depth_file(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        header = f.read(56)
        version, symbol, reserved, num_records = struct.unpack('=i32s16si', header)
        
        data = []
        for _ in range(num_records):
            record = f.read(56)
            timestamp, price, size, side, position, is_end_of_refresh = struct.unpack('=qddifi', record)
            
            # Convert timestamp to datetime
            dt = datetime(1899, 12, 30) + timedelta(days=timestamp)
            
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
    # Sort data by timestamp and side
    df = df.sort_values(['timestamp', 'side', 'price'])
    
    # Group by timestamp and side, then take top 'levels' rows for each group
    grouped = df.groupby(['timestamp', 'side'])
    bids = grouped.apply(lambda x: x[x['side'] == 'bid'].head(levels)).reset_index(drop=True)
    asks = grouped.apply(lambda x: x[x['side'] == 'ask'].head(levels)).reset_index(drop=True)
    
    # Pivot the data to create separate columns for each level
    bids_pivot = bids.pivot(index='timestamp', columns='position', values=['price', 'size'])
    asks_pivot = asks.pivot(index='timestamp', columns='position', values=['price', 'size'])
    
    # Flatten column names
    bids_pivot.columns = [f'bid_price_{i}' if 'price' in col else f'bid_size_{i}' for i, col in enumerate(bids_pivot.columns, 1)]
    asks_pivot.columns = [f'ask_price_{i}' if 'price' in col else f'ask_size_{i}' for i, col in enumerate(asks_pivot.columns, 1)]
    
    # Combine bid and ask data
    combined = pd.concat([bids_pivot, asks_pivot], axis=1).reset_index()
    
    return combined

def convert_depth_to_parquet(depth_file, parquet_file, levels=10):
    # Read .DEPTH file
    df = read_depth_file(depth_file)
    
    # Process depth data
    processed_df = process_depth_data(df, levels)
    
    # Convert to Parquet
    table = pa.Table.from_pandas(processed_df)
    pq.write_table(table, parquet_file)

    print(f"Converted {depth_file} to {parquet_file}")
    print(f"Data shape: {processed_df.shape}")
    print(f"Columns: {processed_df.columns.tolist()}")

# Usage
depth_file = 'path_to_your_file.depth'
parquet_file = 'level2_data.parquet'
convert_depth_to_parquet(depth_file, parquet_file, levels=10)