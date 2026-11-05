
from pathlib import Path
import pandas as pd

# Find supported dataset files in the data folder
data_folder = Path("data")

files = (
    list(data_folder.glob("*.csv"))
    + list(data_folder.glob("*.xlsx"))
    + list(data_folder.glob("*.xls"))
)

if not files:
    raise FileNotFoundError(
        "No CSV or Excel file found. "
        "Download the NSAP dataset into the data folder."
    )

# Use the first available dataset file
file_path = files[0]
print(f"\nDataset file: {file_path}")

# Load CSV or Excel
if file_path.suffix.lower() == ".csv":
    df = pd.read_csv(file_path)
else:
    df = pd.read_excel(file_path)

# Display dataset information
print("\n--- FIRST FIVE ROWS ---")
print(df.head().to_string())

print("\n--- DATASET SHAPE ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes.to_string())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum().to_string())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- UNIQUE VALUES PER COLUMN ---")
for column in df.columns:
    print(f"{column}: {df[column].nunique(dropna=True)}")
