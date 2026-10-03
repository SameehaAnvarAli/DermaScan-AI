import os

import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = "data/splits/train.csv"

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# ============================================================
# Calculate class weights
# ============================================================

def calculate_class_weights():

    df = pd.read_csv(TRAIN_CSV)

    class_counts = df["dx"].value_counts()

    total_samples = len(df)
    num_classes = len(CLASS_NAMES)

    weights = []

    for class_name in CLASS_NAMES:

        count = class_counts[class_name]

        weight = total_samples / (
            num_classes * count
        )

        weights.append(weight)

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    return weights


# ============================================================
# Create weighted loss function
# ============================================================

def create_loss_function():

    class_weights = calculate_class_weights()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    return criterion, class_weights


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DermaScan AI - Class Weighted Loss Test")
    print("=" * 60)

    criterion, class_weights = create_loss_function()

    print("\nClass weights:")

    for class_name, weight in zip(
        CLASS_NAMES,
        class_weights
    ):
        print(
            f"{class_name:6s} : {weight.item():.4f}"
        )

    print("\nLoss function:")
    print(criterion)

    # Test with fake model outputs
    fake_outputs = torch.randn(4, 7)

    fake_labels = torch.tensor([
        0,
        1,
        4,
        5
    ])

    loss = criterion(
        fake_outputs,
        fake_labels
    )

    print("\nTest loss:", loss.item())

    print("\nClass-weighted loss test completed successfully!")