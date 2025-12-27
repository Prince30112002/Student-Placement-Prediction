# src/data_preprocessing.py

import pandas as pd
import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'student_placement.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'student_placement_processed.csv')

# ---------- Load Data ----------
try:
    df = pd.read_csv(RAW_DATA_PATH)
    print("Raw data loaded successfully!")
except FileNotFoundError:
    print(f"Error: File not found at {RAW_DATA_PATH}")
    exit()

# ---------- Preview Data ----------
print("\nFirst 5 rows of raw data:")
print(df.head())

# ---------- Data Cleaning ----------
# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.fillna({
    'ssc_p': df['ssc_p'].median(),
    'hsc_p': df['hsc_p'].median(),
    'degree_p': df['degree_p'].median(),
    'etest_p': df['etest_p'].median(),
    'mba_p': df['mba_p'].median(),
    'salary': 0,
    'status': 'Not Placed'
})

# ---------- Feature Engineering ----------
# Create binary target variable
df['Placed'] = df['status'].apply(lambda x: 1 if x.lower() == 'placed' else 0)

# Drop original 'status' column
df = df.drop('status', axis=1)

# ---------- Save Processed Data ----------
os.makedirs(os.path.join(BASE_DIR, 'data', 'processed'), exist_ok=True)
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"\nProcessed data saved successfully at: {PROCESSED_DATA_PATH}")

# ---------- Summary ----------
print("\nData Summary:")
print(df.info())
print("\nFirst 5 rows of processed data:")
print(df.head())
