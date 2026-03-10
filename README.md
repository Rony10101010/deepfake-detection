# Deepfake Detection

This project detects whether a face image is **real** or **fake** using **VGG16** and **Grad-CAM**.

## Dataset
140k Real and Fake Faces

## Environment
- Google Colab
- Kaggle API
- Google Drive

## Model
- VGG16 pretrained on ImageNet
- Binary classification:
  - real
  - fake

## Project Structure

```text
deepfake-detection/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── explainability/
│   └── inference/
├── outputs/
│   ├── models/
│   ├── figures/
│   └── reports/
├── .gitignore
├── README.md
└── requirements.txt