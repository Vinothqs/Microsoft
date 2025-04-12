import pandas as pd
import numpy as np

# Define input and output files
input_file = "GUIDE_Train.csv"
output_file = "cleaned_GUIDE_Train.parquet"

# Define chunk size (adjust if needed)
chunk_size = 100000

# Columns to optimize
date_cols = ["Timestamp"]
categorical_cols = ["Category", "IncidentGrade", "EntityType", "EvidenceRole", 
                    "ThreatFamily", "ResourceType", "Roles", "AntispamDirection", 
                    "SuspicionLevel", "LastVerdict"]
int_cols = ["Id", "OrgId", "IncidentId", "AlertId", "DeviceId", "Sha256", 
            "IpAddress", "Url", "AccountSid", "AccountUpn", "AccountObjectId", 
            "AccountName", "DeviceName", "NetworkMessageId", "RegistryKey", 
            "RegistryValueName", "RegistryValueData", "ApplicationId", 
            "ApplicationName", "OAuthApplicationId", "FileName", "FolderPath", 
            "ResourceIdName", "OSFamily", "OSVersion", "CountryCode", "State", "City"]

# Process file in chunks
chunk_list = []

for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    # Convert timestamp
    chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"], errors="coerce")
    
    # Convert categorical columns
    for col in categorical_cols:
        chunk[col] = chunk[col].astype("category")
    
    # Convert integer columns to reduce memory usage
    for col in int_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce", downcast="integer")

    # Handle missing values (Fill or Drop based on needs)
    chunk.fillna(method="ffill", inplace=True)  # Forward-fill missing values
    
    # Append processed chunk to list
    chunk_list.append(chunk)

# Combine all chunks into a single DataFrame
df_cleaned = pd.concat(chunk_list, ignore_index=True)

# Save cleaned data in a compressed, efficient format
df_cleaned.to_parquet(output_file, index=False)

print(f"Data cleaning completed! File saved as {output_file}")