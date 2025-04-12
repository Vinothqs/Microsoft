import pyarrow.parquet as pq
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# File paths
parquet_file_path = "D:/vinoth/capstone/Microsoft/cleaned_GUIDE_Train.parquet"
csv_temp_file = "D:/vinoth/capstone/Microsoft/temp_data.csv"
model_file = "D:/vinoth/capstone/Microsoft/model.pkl"

# Define chunk size
batch_size = 100000  
target_column = "IncidentGrade"

# Function to preprocess data
def preprocess_data(df):
    # Convert datetime columns to Unix timestamps
    for col in df.select_dtypes(include=["object"]):
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9
        except:
            pass  # Ignore non-datetime columns

    # Convert categorical columns to numerical values
    for col in df.select_dtypes(include=["object", "category"]):
        df[col] = df[col].astype("category").cat.codes

    return df

# Convert Parquet to CSV in chunks
print("🔄 Converting Parquet to CSV in chunks...")
parquet_file = pq.ParquetFile(parquet_file_path)

with open(csv_temp_file, "w", newline="", encoding="utf-8") as f:
    for i, batch in enumerate(parquet_file.iter_batches(batch_size)):
        chunk = batch.to_pandas()
        chunk = preprocess_data(chunk)  # Apply preprocessing
        chunk.to_csv(f, index=False, header=(i == 0), mode="a")  # Append to CSV

print("✅ Parquet conversion complete!")

# Now process the CSV in chunks
model = RandomForestClassifier(n_estimators=10, random_state=42)

print("🚀 Training model in chunks...")
for chunk in pd.read_csv(csv_temp_file, chunksize=batch_size):
    print(f"Processing chunk with {len(chunk)} rows...")

    chunk = preprocess_data(chunk)  # Ensure numeric data
    chunk = chunk.dropna(subset=[target_column])  # Drop missing target values

    X = chunk.drop(columns=[target_column])
    y = chunk[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

print("🎉 Model training complete!")

# Save trained model
joblib.dump(model, model_file)
print(f"💾 Model saved successfully at {model_file}!")

# Clean up: Delete temp CSV file
os.remove(csv_temp_file)
print("🗑️ Temporary CSV file deleted.")