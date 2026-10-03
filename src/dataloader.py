import torch
from torch.utils.data import DataLoader

from dataset import HAM10000Dataset, train_transform, val_test_transform


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 16

NUM_WORKERS = 0


# ============================================================
# Create datasets
# ============================================================

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


# ============================================================
# Create DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# Test DataLoaders
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DermaScan AI - DataLoader Test")
    print("=" * 60)

    print("\nBatch size:", BATCH_SIZE)

    print("\nNumber of batches:")
    print("Training   :", len(train_loader))
    print("Validation :", len(val_loader))
    print("Test       :", len(test_loader))

    # Get one batch
    images, labels = next(iter(train_loader))

    print("\nFirst training batch:")
    print("Image tensor shape :", images.shape)
    print("Label tensor shape :", labels.shape)

    print("\nLabels in first batch:")
    print(labels)

    print("\nImage tensor data type:", images.dtype)
    print("Label tensor data type:", labels.dtype)

    print("\nDataLoader test completed successfully!")
    