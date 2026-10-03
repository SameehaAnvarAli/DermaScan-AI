import os
import pandas as pd

# -----------------------------
# Paths
# -----------------------------

DATA_DIR = "data/raw"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")


# -----------------------------
# Check metadata file
# -----------------------------

print("=" * 60)
print("DermaScan AI - Dataset Check")
print("=" * 60)

print("\nChecking metadata file...")

if not os.path.exists(METADATA_PATH):
    print("ERROR: HAM10000_metadata.csv was not found!")
    exit()

print("Metadata file found!")


# -----------------------------
# Load metadata
# -----------------------------

df = pd.read_csv(METADATA_PATH)

print("\nMetadata information")
print("-" * 60)

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns:")
print(list(df.columns))


# -----------------------------
# Missing values
# -----------------------------

print("\nMissing values")
print("-" * 60)

print(df.isnull().sum())


# -----------------------------
# Lesion categories
# -----------------------------

print("\nLesion categories")
print("-" * 60)

print(df["dx"].value_counts())


# -----------------------------
# Check image files
# -----------------------------

print("\nChecking image files")
print("-" * 60)

image_files = {
    os.path.splitext(filename)[0]
    for filename in os.listdir(DATA_DIR)
    if filename.lower().endswith(".jpg")
}

print("JPG files found:", len(image_files))

metadata_image_ids = set(df["image_id"].astype(str))

print("Image IDs in metadata:", len(metadata_image_ids))


# -----------------------------
# Find missing images
# -----------------------------

missing_images = metadata_image_ids - image_files
extra_images = image_files - metadata_image_ids

print("\nImage matching")
print("-" * 60)

print("Metadata images missing from folder:", len(missing_images))
print("Extra JPG files not in metadata:", len(extra_images))


# -----------------------------
# Display examples
# -----------------------------

if missing_images:
    print("\nExample missing images:")
    print(list(missing_images)[:10])

if extra_images:
    print("\nExample extra images:")
    print(list(extra_images)[:10])


# -----------------------------
# Final summary
# -----------------------------

print("\n" + "=" * 60)
print("Dataset check completed!")
print("=" * 60)