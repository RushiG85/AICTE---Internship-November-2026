
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings("default")

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("outputs/nsap_cleaned.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "schemecode"

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run python eda.py first."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

required = [
    "schemecode",
    "statename",
    "lgddistrictcode",
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

missing = set(required) - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if df[required].isnull().any().any():
    raise ValueError(
        "Required columns contain missing values. "
        "Investigate them before training."
    )

if (df["totalbeneficiaries"] <= 0).any():
    raise ValueError(
        "Beneficiary counts must be positive for proportion features."
    )

print("=" * 65)
print("NSAP EXPLORATORY CLASSIFICATION")
print("=" * 65)

print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(df[TARGET].value_counts())

# --------------------------------------------------
# 3. CREATE DISTRICT GROUPS
# --------------------------------------------------

# District codes may repeat across states, so combine both.
groups = (
    df["statename"].astype(str).str.strip()
    + "_"
    + df["lgddistrictcode"].astype(str)
)

print("\nUnique district groups:", groups.nunique())

# --------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------

# Use proportions rather than raw population totals.
# These describe aggregate records, NOT individual applicants.

def create_features(data):
    total = data["totalbeneficiaries"].replace(0, np.nan)

    features = pd.DataFrame(index=data.index)

    # Gender proportions
    features["male_share"] = data["totalmale"] / total
    features["female_share"] = data["totalfemale"] / total
    features["transgender_share"] = (
        data["totaltransgender"] / total
    )

    # Social-category proportions
    features["sc_share"] = data["totalsc"] / total
    features["st_share"] = data["totalst"] / total
    features["general_share"] = data["totalgen"] / total
    features["obc_share"] = data["totalobc"] / total

    # Administrative coverage proportions
    features["aadhaar_coverage"] = data["totalaadhaar"] / total
    features["mobile_coverage"] = data["totalmpbilenumber"] / total

    return features.replace([np.inf, -np.inf], np.nan)

X = create_features(df)
y = df[TARGET].astype(str)

if X.isnull().any().any():
    raise ValueError("Engineered features contain missing values.")

print("\nFeatures used:")
print(X.columns.tolist())

# Confirm every district group has records from each class.
group_class_counts = pd.crosstab(groups, y)
print("\nScheme coverage across district groups:")
print(group_class_counts.notna().all(axis=0).to_string())

# --------------------------------------------------
# 5. GROUP-AWARE CROSS-VALIDATION
# --------------------------------------------------

# Rows from the same district are kept in the same fold.
# This reduces the risk of memorizing district-specific patterns.

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Materialize folds so every model uses identical splits.
splits = list(cv.split(X, y, groups=groups))

for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            f"District leakage detected in fold {fold_number}"
        )

    if y.iloc[train_idx].nunique() < y.nunique():
        raise ValueError(
            f"Fold {fold_number} lacks one or more training classes."
        )

    print(
        f"Fold {fold_number}: "
        f"train={len(train_idx)}, test={len(test_idx)}, "
        f"test districts={len(test_groups)}"
    )

# --------------------------------------------------
# 6. DEFINE CANDIDATE MODELS
# --------------------------------------------------

models = {
    "Majority Baseline": DummyClassifier(
        strategy="prior"
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )),
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    ),
}

# --------------------------------------------------
# 7. CROSS-VALIDATED EVALUATION
# --------------------------------------------------

results = []
predictions = {}

for name, model in models.items():
    print(f"\nEvaluating: {name}")

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=splits,
        method="predict",
        n_jobs=-1,
    )

    predictions[name] = y_pred

    results.append({
        "model": name,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "macro_f1": f1_score(
            y, y_pred, average="macro", zero_division=0
        ),
    })

    print(
        classification_report(
            y,
            y_pred,
            zero_division=0,
        )
    )

results_df = pd.DataFrame(results).sort_values(
    "macro_f1", ascending=False
)

print("\n--- MODEL COMPARISON ---")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)

# --------------------------------------------------
# 8. CONFUSION MATRIX
# --------------------------------------------------

# The baseline is useful for comparison, but only plot the
# two trained classifiers.
class_labels = sorted(y.unique())

for name in ["Logistic Regression", "Random Forest"]:
    cm = confusion_matrix(
        y,
        predictions[name],
        labels=class_labels,
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )

    plt.title(f"Grouped Cross-Validation: {name}")
    plt.xlabel("Predicted Scheme")
    plt.ylabel("Actual Scheme")
    plt.tight_layout()

    filename = (
        name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150,
    )
    plt.close()

# --------------------------------------------------
# 9. SAVE THE BEST EXPLORATORY MODEL
# --------------------------------------------------

# Select using cross-validated macro F1.
# The baseline is never eligible for saving as the final model.
eligible = results_df[
    results_df["model"].isin([
        "Logistic Regression",
        "Random Forest",
    ])
]

best_name = eligible.iloc[0]["model"]
best_model = models[best_name]

best_model.fit(X, y)

joblib.dump(
    best_model,
    MODEL_DIR / "nsap_exploratory_model.joblib",
)

with open(
    MODEL_DIR / "model_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump({
        "target": TARGET,
        "model": best_name,
        "features": X.columns.tolist(),
        "classes": [str(c) for c in best_model.classes_],
        "evaluation": "5-fold StratifiedGroupKFold by district",
        "dataset_rows": len(df),
        "unique_districts": int(groups.nunique()),
        "purpose": "Exploratory district-level aggregate classification only",
        "individual_eligibility_validated": False,
    }, f, indent=2)

print("\nBest exploratory model:", best_name)
print("Saved model:", MODEL_DIR / "nsap_exploratory_model.joblib")
print("Saved metadata:", MODEL_DIR / "model_metadata.json")
print("Saved comparison:", OUTPUT_DIR / "model_comparison.csv")

print("\nIMPORTANT:")
print(
    "These results describe aggregate-record classification only. "
    "They do not validate individual eligibility or establish "
    "generalization to other states or years."
)
