Implement the training pipeline for the deepfake detection project.

Output file:
src/training/train.py

Requirements:

Use the dataset loader module and VGG16 model module.

Training configuration:

batch_size = 32
learning_rate = 1e-4
epochs = 20

Loss function:
CrossEntropyLoss

Optimizer:
Adam

The training pipeline must include:

- training loop
- validation loop
- metric tracking

Training loop must perform:

forward pass
loss calculation
backward pass
optimizer step

Validation loop must compute:

validation loss
validation accuracy

Additional requirements:

- support GPU if available
- print epoch progress
- track train and validation metrics
- save best model checkpoint

Checkpoint path:

outputs/models/best_model.pth

Suggested functions:

train_one_epoch(...)
validate_one_epoch(...)
train_model(...)