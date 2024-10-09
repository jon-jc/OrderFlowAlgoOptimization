import pyarrow.parquet as pq

table = pq.read_table('level2_data.parquet')
print("Columns:", table.column_names)
print("Number of rows:", table.num_rows)
print("First few rows:")
print(table.to_pandas().head())