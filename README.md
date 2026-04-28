# 🔍 Deepfake Face Detection

A face-image deepfake classifier built with PyTorch and VGG16. Classifies an input face image as **real** or **fake**, with optional Grad-CAM heatmap visualization to explain the model's decision.

---

## What This App Does

- Classifies face images as `real` or `fake` using a fine-tuned VGG16 model
- Generates Grad-CAM heatmap overlays to visually explain predictions
- Provides a desktop GUI for non-technical users
- Includes full training, evaluation, and inference pipelines

---

## ✨ Features

- **VGG16 transfer learning** — ImageNet-pretrained backbone with a custom 2-class head
- **Grad-CAM explainability** — highlights the facial regions driving each prediction
- **Confidence scoring** — returns both predicted label and confidence percentage
- **Desktop GUI** — Tkinter-based app with image picker and live results
- **Evaluation metrics** — accuracy, precision, recall, F1, confusion matrix
- **CPU-friendly inference** — runs locally without GPU requirement

---

## 📦 Dataset

[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) — sourced from Kaggle.

---

## ☁️ Training Environment

- Google Colab (GPU runtime)
- Kaggle API for dataset download
- Google Drive for checkpoint storage

---

## 🏗️ Project Structure

```
deepfake-detection/
├── notebooks/
├── src/
│   ├── data/
│   │   └── dataset_loader.py        # Transforms, datasets, dataloaders
│   ├── models/
│   │   └── vgg16_model.py           # VGG16 model definition + feature freezing
│   ├── training/
│   │   └── train.py                 # Training loop, validation, checkpointing
│   ├── evaluation/
│   │   └── evaluate.py              # Metrics, confusion matrix, CLI eval
│   ├── explainability/
│   │   └── gradcam.py               # Grad-CAM forward/backward hooks
│   └── inference/
│       ├── predict.py               # Inference API + CLI
│       └── desktop_app.py           # Tkinter desktop GUI
├── outputs/
│   ├── models/
│   ├── figures/
│   └── reports/
├── .gitignore
├── README.md
└── requirements.txt
```

### Inference Data Flow

```
Input Image (PIL)
      ↓
Validation Transform (normalize + resize)
      ↓
VGG16 Forward Pass
      ↓
(predicted_class, confidence)
      ↓ (optional)
Grad-CAM → (heatmap, overlay)
```

---

## 🛠️ Tech Stack

| Library | Role |
|---------|------|
| `PyTorch` | Deep learning framework |
| `torchvision` | VGG16 pretrained model + transforms |
| `OpenCV` | Heatmap overlay generation |
| `Pillow` | Image loading and preprocessing |
| `scikit-learn` | Evaluation metrics |
| `matplotlib` | Confusion matrix + figure output |
| `NumPy` / `pandas` | Data handling |
| `Tkinter` | Desktop GUI |

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Single image inference (CLI)

```bash
python -m src.inference.predict \
  --checkpoint <path/to/model.pth> \
  --image <path/to/face.jpg> \
  --save-dir outputs/figures
```

### Run desktop GUI

```bash
python -m src.inference.desktop_app --checkpoint <path/to/model.pth>
```

### Evaluate on dataset

```bash
python -m src.evaluation.evaluate \
  --checkpoint <path/to/model.pth> \
  --data-root <dataset_root> \
  --batch-size 32 \
  --num-workers 4
```

### Train the model

```python
from src.training.train import train_model

train_model(...)  # configure args in Python or notebook
```

> **Note:** Input must be face images only. Video input is not supported. Model checkpoint required for inference — not included in this repo.

---

## 🧠 Notable Technical Decisions

- **In-place ReLU disabled** globally in VGG16 to prevent Grad-CAM backward hook conflicts
- **Preprocessing consistency** — inference reuses the same validation transforms as training to prevent normalization mismatch
- **Best checkpoint policy** — saves model only when validation accuracy improves
- **Feature freezing support** — convolutional layers can be frozen for faster fine-tuning

---
## License
MIT License

## 👤 Author

**Rony Dawoud** — Flutter & Mobile Developer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/rony-dawoud)
