Goal: verify the deepfake detection project repository structure.

The repository structure already exists and should be respected.

Repository root:

deepfake-detection/

Structure:

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

Instructions for Codex:

1. Do not change the repository structure.
2. Generate Python modules inside the existing folders only.
3. Do not create unnecessary files.
4. Avoid notebook-based implementations.
5. All logic should be implemented inside the `src/` directory.

The code will be executed in Google Colab by importing modules from `src/`.

Example import in Colab:

import sys
sys.path.append("/content/deepfake-detection/src")

from data.dataset_loader import create_dataloaders