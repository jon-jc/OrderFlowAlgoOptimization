import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

def process_level2_data(csv_file, levels=10):
    df = pd.read_csv(csv_file, parse_dates=['DateTime'])
    df = df.rename(columns={'DateTime': 'timestamp'})
    
    bid_columns = [f'Bid{i}' for i in range(1, levels+1)]
    ask_columns = [f'Ask{i}' for i in range(1, levels+1)]
    bid_size_columns = [f'BidSize{i}' for i in range(1, levels+1)]
    ask_size_columns = [f'AskSize{i}' for i in range(1, levels+1)]
    
    df['bids'] = df[bid_columns].values.tolist()
    df['asks'] = df[ask_columns].values.tolist()
    df['bid_sizes'] = df[bid_size_columns].values.tolist()
    df['ask_sizes'] = df[ask_size_columns].values.tolist()
    
    df['bids'] = df.apply(lambda row: [{'price': p, 'size': s} for p, s in zip(row['bids'], row['bid_sizes']) if pd.notna(p) and pd.notna(s)], axis=1)
    df['asks'] = df.apply(lambda row: [{'price': p, 'size': s} for p, s in zip(row['asks'], row['ask_sizes']) if pd.notna(p) and pd.notna(s)], axis=1)
    
    return df[['timestamp', 'bids', 'asks']]

def convert_level2_to_parquet(csv_file, parquet_file, levels=10):
    df = process_level2_data(csv_file, levels)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_file)

# Usage
convert_level2_to_parquet('path_to_your_level2_csv.csv', 'level2_data.parquet', levels=10)