import pandas as pd
import os

# Paths
RAW_DATA_PATH = "data/raw/student_placement.csv"
PROCESSED_DATA_DIR = "data/processed"
PROCESSED_DATA_PATH = "data/processed/cleaned_student_placement.csv"

# Create processed folder if not exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load raw data
df = pd.read_csv(RAW_DATA_PATH)
print("Raw data loaded successfully!\n")

print("First 5 rows of raw data:")
print(df.head())

# -------------------------
# Data Cleaning
# -------------------------

# Drop salary column (NaN for Not Placed students)
if "salary" in df.columns:
    df.drop(columns=["salary"], inplace=True)

# Handle missing values (if any)
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert target column
df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})

# Save cleaned data
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"\nCleaned data saved at: {PROCESSED_DATA_PATH}")
