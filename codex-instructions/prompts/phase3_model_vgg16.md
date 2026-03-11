Implement the VGG16 model module for the deepfake detection project.

Output file:
src/models/vgg16_model.py

Requirements:

1. Use torchvision.models.vgg16 pretrained on ImageNet.
2. Replace the classifier head to support binary classification (2 classes).
3. Add dropout to reduce overfitting.
4. Provide a clean function to create the model.

Model structure:

VGG16 backbone
Fully connected layer
Dropout
Output layer with 2 units

Training strategy:

- Freeze convolutional layers initially
- Train only the classifier head
- Allow optional fine-tuning of top layers later

Required API:

create_model(num_classes=2, freeze_features=True)

The function should:

- load pretrained VGG16
- modify the classifier
- optionally freeze backbone layers
- return the model ready for training

Code requirements:

- modular
- documented
- reusable in Colab
- avoid training logic in this file