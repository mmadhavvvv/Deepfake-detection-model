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

# ---------------- MODEL INITIALIZATION ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_cache = None

def get_model():
    """Lazy loader for the model to ensure weights are loaded only once."""
    global _model_cache
    if _model_cache is None:
        from models.model_architecture import DeepfakeDetector
        _model_cache = DeepfakeDetector().to(DEVICE)
        
        # Load weights
        weights_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "best_deepfake_model.pth")
        if os.path.exists(weights_path):
            print(f"✅ Loading model weights from: {weights_path}")
            _model_cache.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        else:
            print(f"⚠️ Warning: No trained weights found at {weights_path}. Using random weights.")
            
        _model_cache.eval()
    return _model_cache

# Initialize global Grad-CAM utility
_grad_cam_cache = None

def get_grad_cam():
    global _grad_cam_cache
    model = get_model()
    if _grad_cam_cache is None:
        from src.utils.grad_cam import GradCAM
        # Target the last layer of ResNet-18
        target_layer = model.resnet.layer4[-1]
        _grad_cam_cache = GradCAM(model, target_layer)
    return _grad_cam_cache




# ---------------- TRANSFORMS ----------------
# MUST MATCH TRAINING RESOLUTION (224x224 for High-Res Mode)
IMG_SIZE = 224


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
    
    # Enable gradients FOR GRAD-CAM logic
    face_tensor.requires_grad = True

    # 3. Inference
    model = get_model()
    
    # We MUST enable grad here even if model is in eval() for Grad-CAM to work
    with torch.set_grad_enabled(True):
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