# 🛡️ Deepfake Detection Model: Neural Forensics Lab

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.53.1-FF4B4B.svg)](https://streamlit.io/)

A professional-grade, AI-powered forensic tool designed to detect manipulated facial images with high precision. This project leverages **Deep Learning** and **Explainable AI (XAI)** to provide transparent and reliable binary classification (Real vs. Fake).

---

## ✨ Key Features

*   **🔍 Forensic-Level Detection**: Optimized ResNet-18 architecture trained on a massive corpus to identify GAN and Diffusion-based artifacts.
*   **🔥 Grad-CAM Explainability**: Visualizes "Manipulation Hotspots" using gradient-weighted class activation mapping, showing exactly where the model detects inconsistencies.
*   **🌑 Futuristic Dark UI**: A premium, glassmorphism-inspired dark mode interface built for a seamless forensic experience.
*   **🧬 Spatial Feature Extraction**: Utilizes advanced facial ROI detection and cropping for focused analysis.
*   **⚡ Optimized Performance**: Features a "Laptop Mode" for efficient CPU-based inference and training.

---

## 🚀 Tech Stack

- **Deep Learning Framework**: [PyTorch](https://pytorch.org/)
- **Architecture**: ResNet-18 (Weights: IMAGENET1K_V1)
- **Computer Vision**: OpenCV, PIL
- **Explainable AI**: Grad-CAM Implementation
- **Dashboard**: [Streamlit](https://streamlit.io/) with custom CSS
- **Visualization**: Matplotlib (Agg backend), Seaborn
- **Dataset**: Kaggle 140K Real/Fake Faces Dataset

---

## 🛠️ Project Structure

```text
├── app/
│   └── streamlit_app.py      # Premium Dark Mode Dashboard
├── models/
│   ├── model_architecture.py # ResNet-18 Core Definition
│   └── best_deepfake_model.pth # Trained Model Weights
├── src/
│   ├── inference/
│   │   └── predict.py        # Prediction Engine & Grad-CAM Logic
│   ├── training/
│   │   └── train.py          # Automated Training & Auto-Push Script
│   ├── preprocessing/
│   │   └── face_detection.py # Haar Cascade ROI Extraction
│   └── utils/
│       └── grad_cam.py       # Heatmap Generation Utility
├── requirements.txt          # Project Dependencies
└── README.md                 # Project Documentation
```

---

## 🏁 Getting Started

### 1. Requirements
Ensure you have Python 3.8+ installed.

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/mmadhavvvv/Deepfake-detection-model.git
cd Deepfake-detection-model
pip install -r requirements.txt
```

### 3. Launch the Dashboard
Run the following command to start the forensic laboratory:
```bash
streamlit run app/streamlit_app.py
```
*Access the terminal at `http://localhost:8501`.*

---

## 🔬 Scientific Methodology

### Training Scale
The model is trained on the **Kaggle 140k Real and Fake Faces** dataset. The latest iteration utilizes **40,000 balanced images** to ensure robust generalization across diverse demographics and lighting conditions.

### Explainability
Unlike "black-box" models, this system provides a **Grad-CAM heatmap**. It highlights synthetic structural inconsistencies in regions like the skin texture, eyes, and facial boundaries where neural generators often leave subtle artifacts.

---

## 🚧 Ongoing Research
This project is part of a continuous effort to better AI safety and media authenticity. Future updates will include:
- Support for video temporal consistency analysis.
- Multi-facial detection in group settings.
- Integration of Vision Transformers (ViT).

---
*Developed for High-Precision Forensic Validation.*
