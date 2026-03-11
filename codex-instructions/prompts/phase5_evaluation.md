Implement the evaluation module for the trained deepfake detection model.

Output file:
src/evaluation/evaluate.py

Requirements:

Evaluate the trained model using the test dataset.

Metrics to compute:

accuracy
precision
recall
F1 score
confusion matrix

Use scikit-learn for evaluation metrics.

The evaluation module should:

1. Load the trained model checkpoint.
2. Run inference on the test dataset.
3. Compute metrics.
4. Print a classification report.
5. Plot and save a confusion matrix.

Outputs should be saved to:

outputs/reports/
outputs/figures/

Suggested functions:

load_model(...)
evaluate_model(...)
plot_confusion_matrix(...)