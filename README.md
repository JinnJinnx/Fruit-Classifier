# 🍎 Fruit Classifier

A fruit classification system developed for a **Computer Vision project**, using **MobileNetV2 and transfer learning** to identify fruit types from **real-time webcam input or saved images**.

---

## Overview

This project implements a **deep learning-based fruit classification system** that recognizes different fruit types from images.

The model uses **MobileNetV2 pretrained on ImageNet**, enabling efficient feature extraction and accurate classification. The system can process images from both **real-time webcam input** and **saved image files**, making it suitable for interactive computer vision applications.

---

## Features

- Real-time fruit classification using a webcam
- Prediction from saved image files
- Transfer learning with MobileNetV2
- Image preprocessing and feature extraction
- Softmax-based multi-class classification

---

## Model Architecture

The classifier is based on **MobileNetV2**, a lightweight convolutional neural network designed for efficient image classification.

Processing pipeline:

Input Image  
↓  
Image Preprocessing  
↓  
MobileNetV2 Feature Extraction  
↓  
Fully Connected Layer  
↓  
Softmax Classification  
↓  
Predicted Fruit Class  

---

## Dataset

The model was trained using:

- **Fruits-360 dataset**
- Additional fruit images collected manually to improve real-world performance

Images were resized and normalized before being used for training.

---

## Usage

Run the classifier:

python fruit_classifier.py

The program will:

1. Capture or load an image  
2. Preprocess the image  
3. Run inference using the trained model  
4. Display the predicted fruit class  

---

## Project Structure

Fruit-Classifier  
│  
├── fruit_classifier.py  
└── README.md  

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Deep Learning  
- Computer Vision  

---

## Author

Developed as part of a **Computer Vision project**.
