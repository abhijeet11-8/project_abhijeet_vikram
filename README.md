# Emotion Recognition from Facial Images

This repository contains a deep learning-based project for emotion recognition from facial images. The model is trained on grayscale images and classifies them into one of seven emotion categories: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, and `neutral`.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Validation](#validation)
- [Testing](#testing)
- [Model Architecture](#model-architecture)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Custom Model Architecture**: Includes convolutional layers, residual blocks, and dropout for regularization.
- **Focal Loss**: Handles class imbalance effectively.
- **Training Pipeline**: Supports early stopping and learning rate scheduling.
- **Validation and Testing**: Includes metrics like accuracy and loss.
- **Preprocessing**: Grayscale conversion, resizing, and normalization of images.
- **Checkpointing**: Saves the best model during training.
- **Early stopping**: Stops the model ealy to prevent overfitting or too much loss.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/emotion-recognition.git
   cd emotion-recognition
   ```
2. Ensure you have PyTorch installed. Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your system.

---

## Usage

### Training
To train the model, run:
```bash
python interface.py
```

### Testing
To test the model on a dataset, run:
```bash
python predict.py
```

---

## Project Structure

```
.
├──
├── config.py          # Configuration file (hyperparameters, loss functions)
├── dataset.py         # Data loading and preprocessing
├── interface.py       # Main script for training and validation
├── model.py           # Model architecture
├── predict.py         # Script for testing and predictions
├── train.py           # Training utilities
├── test.py            # Testing utilities
├── val.py             # Validation utilities
├── utils.py           # Helper functions (e.g., saving/loading models)
├── final_weights_new.pth  # Pre-trained model weights
└── README.md          # Project documentation
```

---

## Training

1. Place your training data in the `archive/train` directory. The data should be organized into subfolders, one for each emotion class.
2. Run the `interface.py` script to start training:
   ```bash
   python interface.py
   ```

---

## Validation

Validation is performed after each training epoch. The script calculates the validation loss and accuracy, and the learning rate is adjusted using a scheduler based on the validation loss.

---

## Testing

1. Place your test data in the `test` directory or as a zip file named `test.zip`.
2. Run the `predict.py` script to classify the images:
   ```bash
   python predict.py
   ```
3. The script outputs predictions and calculates accuracy.

---

## Model Architecture

The project includes two model architectures:
1. **ImprovedFER2013Model**: A deeper model with residual blocks and global average pooling.
2. **TheModel**: A simpler model with two convolutional layers and fully connected layers.

The default model used is `TheModel`. - both had similar performance but TheModel showed better features reading than the earlier

---

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [FER2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) for inspiration.
- The community for open-source contributions.

---
