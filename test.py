import pandas as pd

# Load the uploaded CSV file
file_path =r"C:\Users\Admin\Documents\GitHub\crypto-signal-bot\Ampli5 KOLs - Online Dashboard (4).csv"  
csv_data = pd.read_csv(file_path)

# Inspect data types to identify potential issues causing "integer out of range"
data_types = csv_data.dtypes

# Convert the "Followers" and "Individual Price" columns to integers within valid range
# if "Followers" in csv_data.columns:
#     csv_data["Followers"] = pd.to_numeric(csv_data["Followers"], errors="coerce").fillna(0).clip(lower=0, upper=2147483647).astype(int)

if "Individual Price" in csv_data.columns:
    csv_data["Individual Price"] = (
        csv_data["Individual Price"]
        .replace({'\$': '', ',': ''}, regex=True)
        .astype(float)
        .fillna(0)
        .clip(lower=0, upper=2147483647)
        .astype(int)
    )

# Save the cleaned file for reference
cleaned_file_path = r"C:\Users\Admin\Documents\GitHub\crypto-signal-bot\test4.csv"  
csv_data.to_csv(cleaned_file_path, index=False)

cleaned_file_path