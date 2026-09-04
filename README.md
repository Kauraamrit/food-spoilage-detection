# 🍎 AI-Based Food Spoilage Detection

An AI-powered image classification system that detects whether an apple, banana, or orange is **fresh or rotten** using Computer Vision and Deep Learning.

## 🎯 Project Overview

Food spoilage can lead to significant household food waste. This project uses a deep learning image classification model to identify the condition of common fruits from an uploaded image.

The system classifies images into six categories:

- 🍎 Fresh Apple
- 🍌 Fresh Banana
- 🍊 Fresh Orange
- 🍎 Rotten Apple
- 🍌 Rotten Banana
- 🍊 Rotten Orange

The trained model is integrated with a **Streamlit web application** where users can upload a fruit image and receive a predicted class with confidence.

## ✨ Features

- Image-based fruit condition classification
- Six fresh/rotten fruit categories
- Transfer learning using MobileNetV2
- Data augmentation for better generalization
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Interactive Streamlit web interface
- Prediction confidence score

## 🧠 Model

The project uses **MobileNetV2** with ImageNet pretrained weights.

### Model Pipeline

```text
Input Image
     ↓
Resize to 224 × 224
     ↓
Data Augmentation
     ↓
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dropout
     ↓
Softmax Classification
     ↓
Predicted Fruit Condition