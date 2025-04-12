import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("model.pkl")

# Define the path to test data
test_file = "GUIDE_Test.csv"

# Load test data
chunk_size = 50000  # Adjust as needed
reader = pd.read_csv(test_file, chunksize=chunk_size)

# Get trained model's expected features
trained_features = model.feature_names_in_
print("\n📌 Model was trained with these features:")
print(list(trained_features))

# Process the test data
output_file = "predictions.csv"
first_write = not os.path.exists(output_file)

# Initialize LabelEncoders for categorical columns
encoders = {}

for chunk in reader:
    print("\n📌 Available columns in test data:")
    print(list(chunk.columns))
    
    # Remove extra columns
    extra_features = set(chunk.columns) - set(trained_features)
    if extra_features:
        print(f"\n⚠️ Extra features in test data (not used in training): {extra_features}")
        chunk = chunk.drop(columns=extra_features, errors='ignore')
    
    # Find missing columns
    missing_features = set(trained_features) - set(chunk.columns)
    if missing_features:
        print(f"\n🚨 Missing required features! Skipping this chunk...")
        continue
    
    # Convert Timestamp to numeric format (UNIX time)
    if 'Timestamp' in chunk.columns:
        chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], errors='coerce').astype('int64') // 10**9
    
    # Encode categorical columns with error handling
    for col in chunk.select_dtypes(include=['object']).columns:
        if col in trained_features:
            if col not in encoders:
                encoders[col] = LabelEncoder()
                encoders[col].fit(chunk[col].astype(str))  # Fit only on current chunk
            chunk[col] = chunk[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
    
    chunk_filtered = chunk[trained_features]
    
    # Make predictions
    predictions = model.predict(chunk_filtered)
    
    # Save predictions
    chunk["Prediction"] = predictions
    chunk.to_csv(output_file, mode='a', index=False, header=first_write)
    first_write = False
    print("🎉 Prediction complete! Predictions saved.")