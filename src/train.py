import os
import torch
import torch.optim as optim
from tqdm import tqdm

from dataloader import train_loader, val_loader
from model import create_model
from loss import create_loss_function


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EPOCHS = 5
LEARNING_RATE = 0.0001

MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training")

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update model weights
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        current_loss = running_loss / total
        current_accuracy = correct / total

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            accuracy=f"{current_accuracy:.4f}"
        )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        progress_bar = tqdm(loader, desc="Validation")

        for images, labels in progress_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            current_loss = running_loss / total
            current_accuracy = correct / total

            progress_bar.set_postfix(
                loss=f"{current_loss:.4f}",
                accuracy=f"{current_accuracy:.4f}"
            )

    validation_loss = running_loss / total
    validation_accuracy = correct / total

    return validation_loss, validation_accuracy


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 60)
    print("DermaScan AI - EfficientNet-B0 Training")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Create models directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = create_model()
    model = model.to(DEVICE)

    print("\nModel loaded successfully.")

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    criterion, class_weights = create_loss_function()
    criterion = criterion.to(DEVICE)

    print("Weighted loss created successfully.")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("AdamW optimizer created successfully.")

    # --------------------------------------------------------
    # Training variables
    # --------------------------------------------------------

    best_validation_accuracy = 0.0

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(NUM_EPOCHS):

        print("\n")
        print("-" * 60)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 60)

        # Training
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # Validation
        validation_loss, validation_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        # Print results
        print("\nEpoch Results:")
        print(f"Training Loss      : {train_loss:.4f}")
        print(f"Training Accuracy  : {train_accuracy:.4f}")
        print(f"Validation Loss    : {validation_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.4f}")

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                },
                BEST_MODEL_PATH
            )

            print("\n✓ Best model saved!")
            print(
                f"✓ Validation Accuracy: "
                f"{validation_accuracy:.4f}"
            )

    # --------------------------------------------------------
    # Training complete
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)

    print(
        f"\nBest Validation Accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    print(f"Best model saved at:")
    print(BEST_MODEL_PATH)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()