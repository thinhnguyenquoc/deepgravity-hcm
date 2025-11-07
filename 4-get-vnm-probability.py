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
# print("Q1:", q1["home_to_ping_distance_category"], q1["distance_category_ping_fraction"])
q3 = combined_data[(combined_data['gadm_name'] == 'Quận 3') & (combined_data['ds'] == '2025-10-01')]
# print("Q3:", q3["home_to_ping_distance_category"], q3["distance_category_ping_fraction"])
q5 = combined_data[(combined_data['gadm_name'] == 'Quận 5') & (combined_data['ds'] == '2025-10-01')]
# print("Q5:", q5["home_to_ping_distance_category"], q5["distance_category_ping_fraction"])
q7 = combined_data[(combined_data['gadm_name'] == 'Quận 7') & (combined_data['ds'] == '2025-10-01')]
# print("Q7:", q7["home_to_ping_distance_category"], q7["distance_category_ping_fraction"])
q10 = combined_data[(combined_data['gadm_name'] == 'Quận 10') & (combined_data['ds'] == '2025-10-01')]
# print("Q10:", q10["home_to_ping_distance_category"], q10["distance_category_ping_fraction"])
q4 = combined_data[(combined_data['gadm_name'] == 'Quận 4') & (combined_data['ds'] == '2025-10-01')]
# print("Q4:", q4["home_to_ping_distance_category"], q4["distance_category_ping_fraction"])

def get_correct_index(place_name, q):
    ob = {
        "place_name": place_name
    }
    for i in range(len(q["home_to_ping_distance_category"])):
        print(q[["home_to_ping_distance_category", "distance_category_ping_fraction"]].iat[i,1])
        if q[["home_to_ping_distance_category", "distance_category_ping_fraction"]].iat[i,0] == "0":
            ob["0"]= float(q[["home_to_ping_distance_category", "distance_category_ping_fraction"]].iat[i,1])
        elif q[["home_to_ping_distance_category", "distance_category_ping_fraction"]].iat[i,0] == "(0, 10)":
            ob["(0, 10)"]= float(q[["home_to_ping_distance_category", "distance_category_ping_fraction"]].iat[i,1])
        elif q[["home_to_ping_distance_category", "distance_category_ping_fraction"]].iat[i,0] == "[10, 100)":
            ob["[10, 100)"]= float(q[["home_to_ping_distance_category", "distance_category_ping_fraction"]].iat[i,1])
        elif q[["home_to_ping_distance_category", "distance_category_ping_fraction"]].iat[i,0] == "100+":
            ob["100+"]= float(q[["home_to_ping_distance_category", "distance_category_ping_fraction"]].iat[i,1])
        else:
            ob["unknown"]= float(q[["home_to_ping_distance_category", "distance_category_ping_fraction"]].iat[i,1])
    return ob

probabilities = []
probabilities.append(get_correct_index("District 1", q1))
probabilities.append(get_correct_index("District 3", q3))
probabilities.append(get_correct_index("District 4", q4))
probabilities.append(get_correct_index("District 5", q5))
probabilities.append(get_correct_index("District 7", q7))
probabilities.append(get_correct_index("District 10", q10))
print(probabilities)

with open("./probability.json", 'w') as f:
    json.dump(probabilities, f, indent=4) # 'indent=4' for pretty-printing