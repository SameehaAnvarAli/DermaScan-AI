import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 7


# ============================================================
# Create EfficientNet-B0
# ============================================================

def create_model():

    print("Loading pretrained EfficientNet-B0...")

    # Load EfficientNet-B0 with ImageNet pretrained weights
    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    # Get the number of input features
    input_features = model.classifier[1].in_features

    # Replace the original classifier
    model.classifier[1] = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# Test model
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DermaScan AI - EfficientNet-B0 Test")
    print("=" * 60)

    # Create model
    model = create_model()

    print("\nModel created successfully!")

    print("\nClassifier:")
    print(model.classifier)

    # Create a fake batch of images
    test_images = torch.randn(
        2,
        3,
        224,
        224
    )

    # Run images through model
    with torch.no_grad():
        outputs = model(test_images)

    print("\nTest input shape:")
    print(test_images.shape)

    print("\nModel output shape:")
    print(outputs.shape)

    print("\nExpected output:")
    print("[batch_size, 7]")

    print("\nEfficientNet-B0 test completed successfully!")