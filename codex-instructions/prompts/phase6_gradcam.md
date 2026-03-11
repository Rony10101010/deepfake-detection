Implement Grad-CAM explainability for the VGG16 deepfake detection model.

Grad-CAM is a mandatory component of the project.

Output file:
src/explainability/gradcam.py

Requirements:

1. Implement Grad-CAM using the last convolutional layer of VGG16.
2. Generate a heatmap showing which regions influenced the prediction.
3. Overlay the heatmap on the original image.

The module must support:

- loading a trained model
- computing Grad-CAM
- visualizing results

Inputs:

- trained model
- input image

Outputs:

- predicted class
- confidence score
- Grad-CAM heatmap
- overlay visualization

Visualization should be saved to:

outputs/figures/

Required functions:

load_model(...)
generate_gradcam(...)
overlay_heatmap(...)
predict_with_gradcam(...)

Code requirements:

- modular
- reusable
- compatible with both Colab experiments and desktop application