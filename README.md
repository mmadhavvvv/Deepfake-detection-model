# Deepfake Detection Model 🕵️‍♂️

A sophisticated spatial-temporal analysis framework designed to detect deepfake video content using Computer Vision and Deep Learning.

## 🚀 Features

- **Spatial-Temporal Analysis**: Combines CNNs (for spatial features) and RNNs (for temporal consistency) to identify subtle manipulation artifacts.
- **Interactive Interface**: Easy-to-use **Streamlit** dashboard for uploading and analyzing videos.
- **Real-time Processing**: Optimized inference pipeline for quick detection.
- **Detailed Insights**: Visual feedback on the probability of a video being a deepfake.

## 🛠️ Tech Stack

- **Frameworks**: [PyTorch](https://pytorch.org/), [OpenCV](https://opencv.org/)
- **UI**: [Streamlit](https://streamlit.io/)
- **Libraries**: NumPy, Pillow, Torchvision
- **Deployment**: Dockerized for consistent environment setup.

## 📁 Structure

- `app/streamlit_app.py`: The web entry point.
- `src/`: Core logic for preprocessing, training, and inference.
- `requirements.txt`: Project dependencies.

## 🏁 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Inference**:
   Upload a video file through the UI and wait for the model to classify it as 'Real' or 'Fake'.

## 🔬 How it Works

The model analyzes frames for physiological inconsistencies (like blink patterns or facial texture anomalies) and temporal flickers that are common in deepfake generation but absent in real footage.

---
*Note: This project is part of ongoing research into AI safety and media authenticity.*
