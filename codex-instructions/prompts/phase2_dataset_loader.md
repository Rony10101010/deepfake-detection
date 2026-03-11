Implement the dataset loading module for the deepfake detection project.

Output file:
src/data/dataset_loader.py

Dataset root in Google Colab:
/content/real_vs_fake/real-vs-fake/

Requirements:
1. Use `torchvision.datasets.ImageFolder`
2. Support:
   - train
   - valid
   - test
3. Create separate transforms for:
   - training
   - validation/test
4. Training transforms:
   - Resize to (224, 224)
   - RandomHorizontalFlip
   - RandomRotation(10)
   - ToTensor
   - Normalize with ImageNet mean/std
5. Validation/test transforms:
   - Resize to (224, 224)
   - ToTensor
   - Normalize with ImageNet mean/std
6. Create configurable DataLoaders:
   - train_loader
   - valid_loader
   - test_loader
7. Shuffle only the training loader
8. Batch size and num_workers must be configurable
9. Print or return:
   - dataset sizes
   - class names
   - class-to-index mapping

Required API:
- function to build transforms
- function to create datasets
- function to create dataloaders

Code requirements:
- modular
- documented
- reusable in Colab notebook imports