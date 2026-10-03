
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Dataset paths
DATA_DIR = "data/raw"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

# Load metadata
df = pd.read_csv(METADATA_PATH)

# Category names
class_names = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesions"
}

# --------------------------------
# 1. Class distribution bar chart
# --------------------------------

class_counts = df["dx"].value_counts()

plt.figure(figsize=(10, 6))
bars = plt.bar(
    [class_names[label] for label in class_counts.index],
    class_counts.values
)

plt.title("HAM10000 - Class Distribution")
plt.xlabel("Lesion Category")
plt.ylabel("Number of Images")
plt.xticks(rotation=35, ha="right")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("reports/class_distribution.png")
plt.show()


# --------------------------------
# 2. Display sample images
# --------------------------------

fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.flatten()

for index, (label, name) in enumerate(class_names.items()):

    sample = df[df["dx"] == label].iloc[0]

    image_path = os.path.join(
        DATA_DIR,
        sample["image_id"] + ".jpg"
    )

    image = Image.open(image_path)

    axes[index].imshow(image)
    axes[index].set_title(name)
    axes[index].axis("off")

# Hide the unused eighth position
axes[7].axis("off")

plt.suptitle("Sample Images from HAM10000")
plt.tight_layout()
plt.savefig("reports/sample_images.png")
plt.show()

print("Visualizations completed successfully!")
print("Saved in the reports folder.")