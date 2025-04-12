import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pyarrow.parquet as pq

# File paths
parquet_file = "D:/vinoth/capstone/Microsoft/cleaned_GUIDE_Train.parquet"
model_file = "D:/vinoth/capstone/Microsoft/model.pkl"
output_file = "D:/vinoth/capstone/Microsoft/evaluation_report.csv"
temp_csv = "D:/vinoth/capstone/Microsoft/temp_eval.csv"

# Convert Parquet to CSV with correct conversions
print("🔄 Converting Parquet to CSV in chunks...")
pq_file = pq.ParquetFile(parquet_file)
batch_size = 100000

with open(temp_csv, "w", newline="", encoding="utf-8") as f:
    for i, batch in enumerate(pq_file.iter_batches(batch_size)):
        chunk = batch.to_pandas()

        # ✅ Convert only 'Timestamp' column to Unix time
        if 'Timestamp' in chunk.columns:
            chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], errors="coerce")
            chunk['Timestamp'] = chunk['Timestamp'].astype("int64") // 10**9

        # ✅ Convert categorical text columns to numeric
        for col in chunk.select_dtypes(include=["object", "category"]):
            chunk[col] = chunk[col].astype("category").cat.codes

        chunk.to_csv(f, index=False, header=(i == 0), mode="a")

print("✅ CSV created successfully.")

# Load model
model = joblib.load(model_file)

# Evaluate model
print("🚀 Evaluating model...")
results = []
for chunk in pd.read_csv(temp_csv, chunksize=batch_size):
    if "IncidentGrade" not in chunk.columns:
        print("⚠️ 'IncidentGrade' column not found. Skipping chunk.")
        continue

    chunk = chunk.dropna(subset=["IncidentGrade"])
    X = chunk.drop(columns=["IncidentGrade"])
    y = chunk["IncidentGrade"]

    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        y_pred = model.predict(X_val)

        report = classification_report(y_val, y_pred, output_dict=True)
        results.append(pd.DataFrame(report).transpose())
    except Exception as e:
        print("⚠️ Skipping chunk due to error:", e)

# Save results
if results:
    final_report = pd.concat(results)
    final_report.to_csv(output_file, index=True)
    print("✅ Evaluation complete. Saved to:", output_file)
else:
    print("🚨 No valid chunks processed.")

# Clean temp file
if os.path.exists(temp_csv):
    os.remove(temp_csv)