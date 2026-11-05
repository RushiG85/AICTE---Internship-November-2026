
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("data/nsap_dataset.csv")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place your CSV file inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("NSAP DATASET: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head().to_string(index=False))

# --------------------------------------------------
# 3. STANDARDIZE COLUMN NAMES
# --------------------------------------------------

df.columns = (
    df.columns.str.strip().str.lower()
)

# --------------------------------------------------
# 4. DATA QUALITY CHECKS
# --------------------------------------------------

print("\n--- DATA TYPES ---")
print(df.dtypes.to_string())

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
print(missing.to_string())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- UNIQUE VALUES ---")
print(df.nunique().to_string())

# Remove exact duplicate records if any exist.
df = df.drop_duplicates().copy()

# Strip whitespace from text columns.
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# Check important columns.
required_columns = [
    "finyear",
    "statename",
    "districtname",
    "schemecode",
    "totalbeneficiaries",
    "totalmale",
    "totalfemale",
    "totaltransgender",
    "totalsc",
    "totalst",
    "totalgen",
    "totalobc",
    "totalaadhaar",
    "totalmpbilenumber",
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

# --------------------------------------------------
# 5. NUMERIC VALIDATION
# --------------------------------------------------

numeric_columns = [
    "totalbeneficiaries",
    "totalmale",
    "totalfemale",
    "totaltransgender",
    "totalsc",
    "totalst",
    "totalgen",
    "totalobc",
    "totalaadhaar",
    "totalmpbilenumber",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

print("\n--- NUMERIC COLUMN SUMMARY ---")
print(df[numeric_columns].describe().round(2).to_string())

# Missing values are reported, not silently converted to zero.
print("\nMissing values after numeric conversion:")
print(df.isnull().sum().to_string())

# --------------------------------------------------
# 6. CHECK DATA CONSISTENCY
# --------------------------------------------------

# Gender totals should match total beneficiaries.
gender_sum = df[
    ["totalmale", "totalfemale", "totaltransgender"]
].sum(axis=1)

gender_mismatch = (
    gender_sum != df["totalbeneficiaries"]
)

# Social-category totals should match total beneficiaries.
social_sum = df[
    ["totalsc", "totalst", "totalgen", "totalobc"]
].sum(axis=1)

social_mismatch = (
    social_sum != df["totalbeneficiaries"]
)

# Counts should not be negative.
negative_counts = (
    df[numeric_columns] < 0
).any(axis=1)

# Contact and Aadhaar counts should not exceed beneficiaries.
aadhaar_exceeds = (
    df["totalaadhaar"] > df["totalbeneficiaries"]
)

mobile_exceeds = (
    df["totalmpbilenumber"] > df["totalbeneficiaries"]
)

print("\n--- DATA CONSISTENCY REPORT ---")
print("Gender-total mismatches:", gender_mismatch.sum())
print("Social-category mismatches:", social_mismatch.sum())
print("Rows containing negative counts:", negative_counts.sum())
print("Aadhaar count exceeds beneficiaries:", aadhaar_exceeds.sum())
print("Mobile count exceeds beneficiaries:", mobile_exceeds.sum())

# Save flagged rows for manual investigation.
flagged = df[
    gender_mismatch
    | social_mismatch
    | negative_counts
    | aadhaar_exceeds
    | mobile_exceeds
].copy()

flagged.to_csv(
    OUTPUT_DIR / "data_quality_issues.csv",
    index=False
)

# --------------------------------------------------
# 7. SCHEME DISTRIBUTION
# --------------------------------------------------

print("\n--- SCHEME DISTRIBUTION ---")

scheme_counts = df["schemecode"].value_counts()
print(scheme_counts.to_string())

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="schemecode",
    order=scheme_counts.index
)

plt.title("Number of Records per NSAP Scheme")
plt.xlabel("Scheme Code")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "scheme_distribution.png",
    dpi=150
)

plt.close()

# --------------------------------------------------
# 8. STATE DISTRIBUTION
# --------------------------------------------------

print("\n--- STATE DISTRIBUTION ---")
print(df["statename"].value_counts().to_string())

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    y="statename",
    order=df["statename"].value_counts().index
)

plt.title("Records by State")
plt.xlabel("Number of Records")
plt.ylabel("State")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "state_distribution.png",
    dpi=150
)

plt.close()

# --------------------------------------------------
# 9. BENEFICIARIES BY SCHEME
# --------------------------------------------------

scheme_summary = (
    df.groupby("schemecode")["totalbeneficiaries"]
    .agg(["count", "mean", "median", "min", "max", "sum"])
    .round(2)
)

print("\n--- BENEFICIARIES BY SCHEME ---")
print(scheme_summary.to_string())

scheme_summary.to_csv(
    OUTPUT_DIR / "scheme_summary.csv"
)

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="schemecode",
    y="totalbeneficiaries"
)

plt.title("Beneficiary Counts by Scheme")
plt.xlabel("Scheme Code")
plt.ylabel("Total Beneficiaries")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "beneficiaries_by_scheme.png",
    dpi=150
)

plt.close()

# --------------------------------------------------
# 10. GENDER DISTRIBUTION
# --------------------------------------------------

gender_summary = (
    df.groupby("schemecode")[
        ["totalmale", "totalfemale", "totaltransgender"]
    ]
    .sum()
)

print("\n--- GENDER DISTRIBUTION BY SCHEME ---")
print(gender_summary.to_string())

gender_summary.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Beneficiary Gender Totals by Scheme")
plt.xlabel("Scheme Code")
plt.ylabel("Number of Beneficiaries")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "gender_distribution.png",
    dpi=150
)

plt.close()

# --------------------------------------------------
# 11. CORRELATION ANALYSIS
# --------------------------------------------------

correlation = df[numeric_columns].corr()

print("\n--- NUMERIC FEATURE CORRELATIONS ---")
print(correlation.round(2).to_string())

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Between Numeric Features")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=150
)

plt.close()

# --------------------------------------------------
# 12. SAVE CLEANED DATA
# --------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "nsap_cleaned.csv",
    index=False
)

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print("Cleaned dataset:", OUTPUT_DIR / "nsap_cleaned.csv")
print("Scheme summary:", OUTPUT_DIR / "scheme_summary.csv")
print("Quality issues:", OUTPUT_DIR / "data_quality_issues.csv")
print("Charts saved in:", OUTPUT_DIR.resolve())
