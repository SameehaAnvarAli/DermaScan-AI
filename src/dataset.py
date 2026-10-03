import os

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data/raw"

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# ============================================================
# Image transformations
# ============================================================

# Training transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation and test transformations
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# HAM10000 Dataset
# ============================================================

class HAM10000Dataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_id = row["image_id"]
        label_name = row["dx"]

        image_path = os.path.join(
            DATA_DIR,
            image_id + ".jpg"
        )

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert class name to numerical label
        label = CLASS_TO_INDEX[label_name]

        return image, label


# ============================================================
# Test the dataset loader
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DermaScan AI - Dataset Loader Test")
    print("=" * 60)

    train_dataset = HAM10000Dataset(
        "data/splits/train.csv",
        transform=train_transform
    )

    val_dataset = HAM10000Dataset(
        "data/splits/val.csv",
        transform=val_test_transform
    )

    test_dataset = HAM10000Dataset(
        "data/splits/test.csv",
        transform=val_test_transform
    )

    print("\nDataset sizes:")
    print("Training   :", len(train_dataset))
    print("Validation :", len(val_dataset))
    print("Test       :", len(test_dataset))

    # Load one training sample
    image, label = train_dataset[0]

    print("\nSample image information:")
    print("Tensor shape :", image.shape)
    print("Label index  :", label)
    print("Label name   :", CLASS_NAMES[label])

    print("\nDataset loader test completed successfully!")