
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = Path("outputs/nsap_cleaned.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

target = "schemecode"
y = df[target].astype(str)

groups = (
    df["statename"].astype(str).str.strip()
    + "_"
    + df["lgddistrictcode"].astype(str)
)

# Recreate the same proportion-based features.
total = df["totalbeneficiaries"].replace(0, np.nan)

X = pd.DataFrame(index=df.index)

X["male_share"] = df["totalmale"] / total
X["female_share"] = df["totalfemale"] / total
X["transgender_share"] = df["totaltransgender"] / total

X["sc_share"] = df["totalsc"] / total
X["st_share"] = df["totalst"] / total
X["general_share"] = df["totalgen"] / total
X["obc_share"] = df["totalobc"] / total

X["aadhaar_coverage"] = df["totalaadhaar"] / total
X["mobile_coverage"] = df["totalmpbilenumber"] / total

# Use identical district-grouped folds for every experiment.
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

splits = list(cv.split(X, y, groups=groups))

# Each experiment removes a different feature group.
experiments = {
    "All features": list(X.columns),

    "Without gender proportions": [
        "sc_share",
        "st_share",
        "general_share",
        "obc_share",
        "aadhaar_coverage",
        "mobile_coverage",
    ],

    "Without gender and social-category proportions": [
        "aadhaar_coverage",
        "mobile_coverage",
    ],

    "Gender proportions only": [
        "male_share",
        "female_share",
        "transgender_share",
    ],
}

results = []
labels = sorted(y.unique())

for experiment, columns in experiments.items():
    features = X[columns]

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    predictions = cross_val_predict(
        model,
        features,
        y,
        cv=splits,
        method="predict",
        n_jobs=-1,
    )

    result = {
        "experiment": experiment,
        "feature_count": len(columns),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y, predictions
        ),
        "macro_f1": f1_score(
            y, predictions,
            average="macro",
            zero_division=0,
        ),
    }

    results.append(result)

    print("\n" + "=" * 60)
    print(experiment)
    print("Features:", columns)
    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions, labels=labels))

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "feature_sensitivity_results.csv",
    index=False,
)

print("\n--- FEATURE SENSITIVITY COMPARISON ---")
print(results_df.round(4).to_string(index=False))

# Check how strongly individual features vary by scheme.
diagnostics = X.copy()
diagnostics["scheme"] = y

print("\n--- FEATURE MEANS BY SCHEME ---")
print(
    diagnostics.groupby("scheme")
    .mean(numeric_only=True)
    .round(3)
    .to_string()
)

print("\nResults saved to:")
print(OUTPUT_DIR / "feature_sensitivity_results.csv")
