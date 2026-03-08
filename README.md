Fruit Classifier

A fruit classification system developed for a Computer Vision project, using MobileNetV2 and transfer learning to identify fruit types from real-time webcam input or saved images.

Overview

This project implements a deep learning-based fruit classification system capable of recognizing different fruit types using image input. The system leverages a MobileNetV2 convolutional neural network pretrained on ImageNet, allowing efficient feature extraction and accurate classification.

The classifier can process images captured from a webcam in real-time or from saved image files, making it suitable for interactive computer vision applications.

Features

Real-time fruit classification using a webcam

Prediction from saved image files

Transfer learning with MobileNetV2

Image preprocessing and feature extraction

Softmax-based multi-class classification

Model Architecture

The model is based on MobileNetV2, a lightweight convolutional neural network designed for efficient image classification tasks.

Key components:

Pretrained backbone: MobileNetV2 (ImageNet weights)

Feature extraction: Convolutional layers learn hierarchical visual patterns such as edges, textures, and shapes

Classifier: Fully connected layer with Softmax activation for multi-class fruit prediction

Dataset

The model was trained using:

Fruits-360 dataset

Additional fruit images captured manually to improve real-world generalization

Images were resized and preprocessed before being used for training.

Training

The model was trained using transfer learning, where pretrained MobileNetV2 weights provide strong low-level feature representations.

Training process:

Freeze the pretrained backbone

Train the classification head

Fine-tune upper layers to improve feature representation

The model learns hierarchical visual features such as:

edges

textures

color patterns

object shapes

Usage

Run the fruit classifier:

python fruit_classifier.py


The system will:

Capture an image from the webcam or load a saved image

Preprocess the image

Pass it through the trained model

Display the predicted fruit type

Project Structure
Fruit-Classifier
│
├── fruit_classifier.py   # Main classification script
├── README.md             # Project documentation

Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Deep Learning

Computer Vision

Author

Developed as part of a Computer Vision project.
