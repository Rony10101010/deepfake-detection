# AGENTS.md

## Project
Deepfake image detection using VGG16.

The system classifies face images into:
- real
- fake

Grad-CAM explainability is a mandatory component of the project.

## Development Workflow

Code is written locally in the GitHub repository and pushed to GitHub.

Google Colab is used only for:
- training
- evaluation
- experiments

The repository contains reusable Python modules under `src/`.

Avoid notebook-heavy implementations unless explicitly requested.

## Dataset

Dataset name:
140k Real and Fake Faces

Runtime dataset path in Google Colab:

/content/real_vs_fake/real-vs-fake/

Expected structure:

/content/real_vs_fake/real-vs-fake/
├── train/
│   ├── real/
│   └── fake/
├── valid/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/

## Project Structure

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
├── README.md
├── requirements.txt
└── .gitignore

## Tech Stack

Language:
Python 3.10+

Framework:
PyTorch

Libraries:
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
Pillow

Environment:
- local development in VS Code
- GitHub for version control
- Google Colab for training and experiments

## Model

Backbone:
VGG16 pretrained on ImageNet

Task:
Binary image classification

Input size:
224 x 224 RGB

## Training Strategy

1. Use transfer learning.
2. Freeze the convolutional backbone initially.
3. Train the classifier head first.
4. Optionally fine-tune upper VGG16 layers later.

## Evaluation Metrics

The evaluation pipeline must compute:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

Metrics should be implemented using scikit-learn.

## Explainability

Grad-CAM must be implemented using the last convolutional layer of VGG16.

Grad-CAM must support:

- generating a heatmap
- overlaying the heatmap on the original image
- saving visualizations to `outputs/figures/`

Grad-CAM must be reusable by:

- evaluation scripts
- inference scripts
- the desktop demo application

## Code Rules

Always:

- write modular reusable Python code
- include docstrings
- use clear function names
- avoid hardcoded paths
- expose configuration values as parameters
- separate responsibilities between modules

Separate modules must exist for:

- data loading
- model definition
- training
- evaluation
- Grad-CAM
- inference

## Colab Integration

Generated `.py` files will be imported into Colab notebooks.

Typical import pattern:

```python
import sys
sys.path.append("/content/deepfake-detection/src")