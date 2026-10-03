import os
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import efficientnet_b0

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import transforms


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

CLASS_LABELS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion"
}

CLASS_INFO = {
    "akiec": "A category containing actinic keratoses and related intraepithelial carcinoma lesions.",
    "bcc": "A category representing basal cell carcinoma lesions.",
    "bkl": "A category of benign keratosis-like skin lesions.",
    "df": "A category representing dermatofibroma lesions.",
    "mel": "A category representing melanoma lesions.",
    "nv": "A category representing melanocytic nevus lesions.",
    "vasc": "A category representing vascular skin lesions."
}

MODEL_PATH = os.path.join(
    "models",
    "best_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = efficientnet_b0(
        weights=None
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        input_features,
        len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(image):

    image = image.convert("RGB")

    image_np = np.array(image)

    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )

    width, height = image.size

    # Resolution
    resolution_ok = (
    width >= 100 and
    height >= 100
    )

    # Blur
    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    blur_ok = blur_score >= 50

    # Brightness
    brightness = np.mean(gray)

    brightness_ok = (
        brightness >= 30 and
        brightness <= 225
    )

    # Contrast
    contrast = np.std(gray)

    contrast_ok = contrast >= 20

    overall_ok = (
        resolution_ok and
        blur_ok and
        brightness_ok and
        contrast_ok
    )

    warnings = []

    if not resolution_ok:
        warnings.append(
            "Image resolution is too low."
        )

    if not blur_ok:
        warnings.append(
            "Image may be too blurry."
        )

    if not brightness_ok:

        if brightness < 30:
            warnings.append(
                "Image appears too dark."
            )
        else:
            warnings.append(
                "Image appears too bright."
            )

    if not contrast_ok:
        warnings.append(
            "Image has low contrast."
        )

    return {
        "quality_ok": overall_ok,
        "width": width,
        "height": height,
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "warnings": warnings
    }


# ============================================================
# PREDICTION + GRAD-CAM
# ============================================================

def predict_image(image, model):

    image = image.convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    predicted_probability = probabilities[
        0,
        predicted_index
    ].item()

    # Grad-CAM
    target_layer = model.features[-1]

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    targets = [
        ClassifierOutputTarget(
            predicted_index
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    resized_image = cv2.resize(
        np.array(image),
        (224, 224)
    )

    display_image = (
        resized_image.astype(
            np.float32
        ) / 255.0
    )

    gradcam_image = show_cam_on_image(
        display_image,
        grayscale_cam,
        use_rgb=True
    )

    # All probabilities
    all_probabilities = {}

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities[0].cpu().numpy()
    ):

        all_probabilities[class_name] = (
            float(probability)
        )

    return {
        "predicted_class": predicted_class,
        "probability": predicted_probability,
        "probabilities": all_probabilities,
        "gradcam": gradcam_image
    }


# ============================================================
# HEADER
# ============================================================

st.title("🔬 DermaScan AI")

st.subheader(
    "Explainable Skin Lesion Screening System"
)

st.write(
    "Upload a skin lesion image to explore an "
    "AI-assisted screening prediction, model "
    "probabilities, and Grad-CAM visualization."
)

st.divider()


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    "⚠️ This is an AI-assisted screening prototype "
    "for educational and research purposes. It is "
    "NOT a medical diagnostic tool. Model predictions "
    "should not be used to diagnose or rule out a "
    "medical condition."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About DermaScan AI")

    st.write(
        "DermaScan AI uses a transfer-learning "
        "EfficientNet-B0 model trained on the "
        "HAM10000 dermatology image dataset."
    )

    st.write(
        "**Model:** EfficientNet-B0"
    )

    st.write(
        "**Classes:** 7"
    )

    st.write(
        "**Explainability:** Grad-CAM"
    )

    st.write(
        "**Framework:** PyTorch"
    )

    st.write(
        "**Interface:** Streamlit"
    )

    st.divider()

    st.caption(
        "Model probability represents model output, "
        "not the probability that a person has a "
        "disease."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Image")

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )


    # --------------------------------------------------------
    # Image quality
    # --------------------------------------------------------

    quality = check_image_quality(
        image
    )

    with col2:

        st.subheader(
            "Image Quality Check"
        )

        st.write(
            f"**Resolution:** "
            f"{quality['width']} × "
            f"{quality['height']}"
        )

        st.write(
            f"**Blur score:** "
            f"{quality['blur_score']:.2f}"
        )

        st.write(
            f"**Brightness:** "
            f"{quality['brightness']:.2f}"
        )

        st.write(
            f"**Contrast:** "
            f"{quality['contrast']:.2f}"
        )

        if quality["quality_ok"]:

            st.success(
                "Image passed the basic technical "
                "quality checks."
            )

        else:

            st.error(
                "Image may not be suitable for "
                "reliable model processing."
            )

            for warning in quality["warnings"]:

                st.write(
                    f"⚠️ {warning}"
                )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if quality["quality_ok"]:

        st.divider()

        with st.spinner(
            "Analyzing image..."
        ):

            model = load_model()

            result = predict_image(
                image,
                model
            )


        predicted_class = (
            result["predicted_class"]
        )

        probability = (
            result["probability"]
        )

        # Confidence level
        if probability < 0.50:

            confidence = "Low"

        elif probability < 0.75:

            confidence = "Moderate"

        else:

            confidence = "Higher"


        # ----------------------------------------------------
        # Prediction result
        # ----------------------------------------------------

        st.subheader(
            "AI Screening Result"
        )

        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )

        with result_col1:

            st.metric(
                "Predicted Category",
                CLASS_LABELS[
                    predicted_class
                ]
            )

        with result_col2:

            st.metric(
                "Model Probability",
                f"{probability * 100:.2f}%"
            )

        with result_col3:

            st.metric(
                "Confidence Level",
                confidence
            )


        # ----------------------------------------------------
        # Educational information
        # ----------------------------------------------------

        st.subheader(
            "Category Information"
        )

        st.info(
            CLASS_INFO[
                predicted_class
            ]
        )


        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        st.subheader(
            "Grad-CAM Explainability"
        )

        st.write(
            "The visualization below highlights image "
            "regions that contributed to the model's "
            "prediction. It represents model behavior "
            "and is not proof of a medically significant "
            "region."
        )

        st.image(
            result["gradcam"],
            caption=(
                "Grad-CAM visualization"
            ),
            use_container_width=True
        )


        # ----------------------------------------------------
        # Class probabilities
        # ----------------------------------------------------

        st.subheader(
            "Class Probabilities"
        )

        probability_data = {
            CLASS_LABELS[class_name]:
                probability_value * 100
            for class_name, probability_value
            in result["probabilities"].items()
        }

        st.bar_chart(
            probability_data
        )


        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        st.subheader(
            "Important Note"
        )

        if confidence == "Low":

            st.warning(
                "The model has limited confidence in "
                "this prediction. The result should not "
                "be interpreted as a diagnosis."
            )

        else:

            st.info(
                "The model produced this screening "
                "prediction, but model output alone "
                "cannot establish a medical diagnosis."
            )

        st.write(
            "If a skin lesion is new, changing, "
            "bleeding, painful, or otherwise concerning, "
            "seek evaluation from a qualified healthcare "
            "professional."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DermaScan AI • AI-assisted skin lesion screening "
    "prototype • Not a medical diagnostic device"
)