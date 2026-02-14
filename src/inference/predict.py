import torch
import cv2
import numpy as np
from torchvision import transforms
import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.model_architecture import DeepfakeDetector
from src.utils.grad_cam import GradCAM, overlay_heatmap

# ---------------- DEVICE ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
def load_trained_model(model_path=None):
    if model_path is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_path = os.path.join(root, "models", "best_deepfake_model.pth")
    
    model = DeepfakeDetector()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_trained_model()

# Initialize GradCAM lazily
grad_cam = None

def get_grad_cam():
    global grad_cam
    if grad_cam is None:
        # ResNet-18 last conv layer is layer4
        target_layer = model.resnet.layer4[-1]
        grad_cam = GradCAM(model, target_layer)
    return grad_cam


# ---------------- TRANSFORMS ----------------
# MUST MATCH TRAINING RESOLUTION (128x128 for Laptop Mode)
IMG_SIZE = 128

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- PREDICTION FUNCTION ----------------
def predict_face(face_bgr):
    """
    Inputs: 
        face_bgr: Image array from cv2.imread
    Returns:
        label (str): Real / Deepfake / Uncertain
        confidence (float): The percentage of certainty
        heatmap_img (np.array): Image with Grad-CAM overlay (RGB)
    """
    # 1. Convert BGR (OpenCV) to RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize for consistent Grad-CAM overlay
    face_rgb_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))

    # 2. Preprocess
    face_tensor = transform(face_rgb_resized).unsqueeze(0).to(DEVICE)
    face_tensor.requires_grad = True


    # 3. Inference
    logits = model(face_tensor)
    score = torch.sigmoid(logits).item()

    # 4. Heatmap Generation
    gc = get_grad_cam()
    heatmap = gc.generate_heatmap(face_tensor)
    heatmap_img = overlay_heatmap(face_rgb_resized, heatmap)


    # 5. Logic
    if score > 0.65:
        label = "DEEPFAKE"
        confidence = score * 100
    elif score < 0.35:
        label = "REAL"
        confidence = (1 - score) * 100
    else:
        label = "UNCERTAIN"
        confidence = score * 100

    return label, confidence, heatmap_img


# ---------------- EXAMPLE USAGE ----------------
if __name__ == "__main__":
    test_image_path = "path_to_your_test_image.jpg"
    image = cv2.imread(test_image_path)
    
    if image is not None:
        result_label, result_conf = predict_face(image)
        print("-" * 30)
        print(f"RESULT: {result_label}")
        print(f"CONFIDENCE: {result_conf:.2f}%")
        print("-" * 30)
    else:
        print("❌ Could not load image. Check path!")