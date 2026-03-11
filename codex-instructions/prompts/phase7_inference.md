Implement the inference module for the deepfake detection project.

Output file:
src/inference/predict.py

Requirements:

1. Load a trained model checkpoint.
2. Preprocess an input image.
3. Run prediction.
4. Compute Grad-CAM visualization.

Output information:

- predicted class (real or fake)
- confidence score
- Grad-CAM heatmap

The module must reuse the Grad-CAM implementation from:

src/explainability/gradcam.py

Suggested functions:

load_trained_model(...)
preprocess_image(...)
predict_image(...)
predict_with_gradcam(...)

Important:

This module will be used both in:

- Google Colab experiments
- the final desktop demo application

Do not include GUI code here.