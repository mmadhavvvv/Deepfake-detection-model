import os
import sys

# Path fix
root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(root)

print("Testing imports...")
try:
    import streamlit as st
    print("Streamlit imported")
    import cv2
    print("CV2 imported")
    import torch
    print("Torch imported")
    from src.preprocessing.face_detection import detect_and_crop_face
    print("Face detection imported")
    from src.inference.predict import predict_face
    print("Predict face imported")
    print("All imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
