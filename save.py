import joblib
import os

# Define the model file path
model_file = "D:/vinoth/capstone/Microsoft/model.pkl"

# Try to get the trained model from memory
try:
    model  # Check if model exists in memory
    joblib.dump(model, model_file)
    print(f"💾 Model saved as {model_file}")
except NameError:
    print("❌ No trained model found in memory! Run model.py first.")