import pyarrow.parquet as pq

# Check Level 1 data
level1_table = pq.read_table('level1_data.parquet')
print("Level 1 Data Columns:", level1_table.column_names)
print("Level 1 Data Shape:", level1_table.num_rows, "rows,", level1_table.num_columns, "columns")

# Check Level 2 data
level2_table = pq.read_table('level2_data.parquet')
print("Level 2 Data Columns:", level2_table.column_names)
print("Level 2 Data Shape:", level2_table.num_rows, "rows,", level2_table.num_columns, "columns")