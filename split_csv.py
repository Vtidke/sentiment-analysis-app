import pandas as pd
import os

# Path to your large CSV (same folder as this script)
file_path = "Reviews.csv"

# Folder to save smaller CSVs inside your project
output_folder = "data/csv_parts"
os.makedirs(output_folder, exist_ok=True)  # automatically creates folders if they don't exist

# Number of rows per smaller CSV (adjust to get multiple files)
rows_per_chunk = 50000  # smaller chunks ensure multiple parts

# Read CSV in chunks and save each chunk
for i, chunk in enumerate(pd.read_csv(file_path, chunksize=rows_per_chunk)):
    output_file = os.path.join(output_folder, f"Reviews_part{i+1}.csv")
    chunk.to_csv(output_file, index=False)
    print(f"Saved {output_file} ({len(chunk)} rows)")

print("\nAll parts saved successfully in 'data/csv_parts/'!")