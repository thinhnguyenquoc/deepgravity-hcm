import json
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
combined_data = read_multiple_csv_files('data/vn', '*.csv')
q1 = combined_data[(combined_data['gadm_name'] == 'Quận 1') & (combined_data['ds'] == '2025-10-01')]
print("Q1:", q1["home_to_ping_distance_category"], q1["distance_category_ping_fraction"])
q3 = combined_data[(combined_data['gadm_name'] == 'Quận 3') & (combined_data['ds'] == '2025-10-01')]
print("Q3:", q3["home_to_ping_distance_category"], q3["distance_category_ping_fraction"])
q5 = combined_data[(combined_data['gadm_name'] == 'Quận 5') & (combined_data['ds'] == '2025-10-01')]
print("Q5:", q5["home_to_ping_distance_category"], q5["distance_category_ping_fraction"])
q7 = combined_data[(combined_data['gadm_name'] == 'Quận 7') & (combined_data['ds'] == '2025-10-01')]
print("Q7:", q7["home_to_ping_distance_category"], q7["distance_category_ping_fraction"])
q10 = combined_data[(combined_data['gadm_name'] == 'Quận 10') & (combined_data['ds'] == '2025-10-01')]
print("Q10:", q10["home_to_ping_distance_category"], q10["distance_category_ping_fraction"])

probabilities = [
    {
        "place_name": "District 1",
        "0": 0.338331,
        "(0, 10)": 0.619545,
        "[10, 100)": 0.034696,
        "100+": 0.007459
    },
    {
        "place_name": "District 3",
        "0": 0.356897,
        "(0, 10)": 0.614933,
        "[10, 100)": 0.023094,
        "100+": 0.005119
    },
    {
        "place_name": "District 5",
        "0": 0.368877,
        "(0, 10)": 0.598244,
        "[10, 100)": 0.026583,
        "100+": 0.006394
    },
    {
        "place_name": "District 7",
        "0": 0.351673,
        "(0, 10)": 0.613202,
        "[10, 100)": 0.032559,
        "100+": 0.002576
    },
    {
        "place_name": "District 10",
        "0": 0.356323,
        "(0, 10)": 0.616555,
        "[10, 100)": 0.035387,
        "100+": 0.023637
    }

]
with open("./probability.json", 'w') as f:
    json.dump(probabilities, f, indent=4) # 'indent=4' for pretty-printing

