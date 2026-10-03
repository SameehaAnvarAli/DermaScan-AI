import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data/raw"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

SPLIT_DIR = "data/splits"


# ============================================================
# Create output folder
# ============================================================

os.makedirs(SPLIT_DIR, exist_ok=True)


# ============================================================
# Load metadata
# ============================================================

print("=" * 70)
print("DermaScan AI - Creating Dataset Splits")
print("=" * 70)

df = pd.read_csv(METADATA_PATH)

print("\nTotal images:", len(df))
print("Total unique lesions:", df["lesion_id"].nunique())


# ============================================================
# Check that each lesion has only one diagnosis
# ============================================================

lesion_label_counts = df.groupby("lesion_id")["dx"].nunique()

if lesion_label_counts.max() > 1:
    print("\nERROR: Some lesions have multiple diagnosis labels.")
    print(lesion_label_counts[lesion_label_counts > 1])
    raise ValueError("Inconsistent lesion labels found.")

print("\nLesion label check: PASSED")


# ============================================================
# First split:
# Approximately 15% TEST
# ============================================================

sgkf_test = StratifiedGroupKFold(
    n_splits=7,
    shuffle=True,
    random_state=42
)

train_val_indices, test_indices = next(
    sgkf_test.split(
        df,
        y=df["dx"],
        groups=df["lesion_id"]
    )
)

train_val_df = df.iloc[train_val_indices].copy()
test_df = df.iloc[test_indices].copy()


# ============================================================
# Second split:
# Approximately 15% VALIDATION
# from the remaining data
# ============================================================

sgkf_val = StratifiedGroupKFold(
    n_splits=6,
    shuffle=True,
    random_state=42
)

train_indices, val_indices = next(
    sgkf_val.split(
        train_val_df,
        y=train_val_df["dx"],
        groups=train_val_df["lesion_id"]
    )
)

train_df = train_val_df.iloc[train_indices].copy()
val_df = train_val_df.iloc[val_indices].copy()


# ============================================================
# Save splits
# ============================================================

train_path = os.path.join(SPLIT_DIR, "train.csv")
val_path = os.path.join(SPLIT_DIR, "val.csv")
test_path = os.path.join(SPLIT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)


# ============================================================
# Display split information
# ============================================================

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

print(f"\nTraining images   : {len(train_df)}")
print(f"Validation images : {len(val_df)}")
print(f"Test images       : {len(test_df)}")

print("\nTraining percentage   : {:.2f}%".format(
    len(train_df) / len(df) * 100
))

print("Validation percentage : {:.2f}%".format(
    len(val_df) / len(df) * 100
))

print("Test percentage       : {:.2f}%".format(
    len(test_df) / len(df) * 100
))


# ============================================================
# Unique lesion counts
# ============================================================

print("\nUnique lesions:")

print("Training lesions   :", train_df["lesion_id"].nunique())
print("Validation lesions :", val_df["lesion_id"].nunique())
print("Test lesions       :", test_df["lesion_id"].nunique())


# ============================================================
# Verify no lesion leakage
# ============================================================

train_lesions = set(train_df["lesion_id"])
val_lesions = set(val_df["lesion_id"])
test_lesions = set(test_df["lesion_id"])

train_val_overlap = train_lesions & val_lesions
train_test_overlap = train_lesions & test_lesions
val_test_overlap = val_lesions & test_lesions

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

print("Train ↔ Validation overlap:", len(train_val_overlap))
print("Train ↔ Test overlap      :", len(train_test_overlap))
print("Validation ↔ Test overlap :", len(val_test_overlap))


# ============================================================
# Class distribution
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print("\nTraining:")
print(train_df["dx"].value_counts().sort_index())

print("\nValidation:")
print(val_df["dx"].value_counts().sort_index())

print("\nTest:")
print(test_df["dx"].value_counts().sort_index())


# ============================================================
# Final message
# ============================================================

print("\n" + "=" * 70)
print("Dataset splitting completed successfully!")
print("=" * 70)

print("\nFiles created:")

print("data/splits/train.csv")
print("data/splits/val.csv")
print("data/splits/test.csv")