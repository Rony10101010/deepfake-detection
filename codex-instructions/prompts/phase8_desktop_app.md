Implement a Windows desktop demo application for the deepfake detection project.

Output file:
src/inference/desktop_app.py

The desktop demo must include Grad-CAM visualization.

Requirements:

Use Tkinter to build the GUI.

The app must allow the user to:

1. Select an image from disk.
2. Display the image.
3. Run deepfake detection.

The app must display:

- prediction (real or fake)
- confidence score
- Grad-CAM heatmap visualization

The application must reuse:

src/inference/predict.py
src/explainability/gradcam.py

Design goals:

- simple user interface
- clear visualization
- compatible with PyInstaller packaging

Expected workflow:

User opens application
→ selects image
→ model predicts real/fake
→ Grad-CAM heatmap is displayed
→ result is shown.