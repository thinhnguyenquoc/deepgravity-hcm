import pandas as pd
import glob
import os

def read_multiple_csv_files(directory_path='./', pattern='*.csv'):
    """
    Read multiple CSV files from a directory and combine them into one DataFrame
    """
    # Get list of CSV files
    search_pattern = os.path.join(directory_path, pattern)
    files = glob.glob(search_pattern)
    
    if not files:
        print("No CSV files found!")
        return None
    
    dfs = []
    successful_files = 0
    
    for file in files:
        try:
            print(f"Reading {file}...")
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file)
            dfs.append(df)
            successful_files += 1
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Successfully read {successful_files} files. Total rows: {len(combined_df)}")
        return combined_df
    else:
        print("No files were successfully read.")
        return None

# Usage
combined_data = read_multiple_csv_files('data/movement-distribution-maps-2025-09-01-2025-10-01', '*.csv')
# print("combined_data", combined_data[(combined_data["country"] == 'VNM') & ("VNM.8" in combined_data["gadm_id"]) ])
filtered_df = combined_data[combined_data["country"] == 'VNM']
filtered_df.to_csv('VNM.csv', index=False)