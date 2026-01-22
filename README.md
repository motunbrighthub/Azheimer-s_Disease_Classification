Alzheimer’s Disease Classification Using MRI Images
 Project Overview

This project focuses on classifying Alzheimer’s disease stages from brain MRI scan images using deep learning. The goal is to build a multi-class image classification model that can distinguish between different stages of Alzheimer’s disease to support early diagnosis and research.

The model classifies MRI images into four categories:

Non-Demented

Very Mild Demented

Mild Demented

Moderate Demented

 Dataset Description

The dataset consists of MRI brain scan images organized into class-specific folders.

Classes & Image Count:

NonDemented: 12,800

VeryMildDemented: 11,200

MildDemented: 10,000

ModerateDemented: 10,000

Although the dataset is not perfectly balanced, class imbalance was handled during training using appropriate techniques (e.g., class weighting).

Source: Kaggle – Alzheimer’s Multiclass Dataset (Equal & Augmented)

 Exploratory Data Analysis (EDA)

Basic EDA was performed to:

Inspect class distribution

Verify image quality and consistency

Confirm folder structure and labels

 Model & Approach

Model Type: Convolutional Neural Network (CNN)

Technique: Transfer Learning

Base Model: Pretrained CNN (fine-tuned)

Loss Function: Categorical Crossentropy

Evaluation Metrics:

Accuracy

Precision

Recall

 ##  Model Performance

The model was trained using transfer learning with fine-tuning and evaluated using multiple performance metrics.

### Best Validation Results
- **Validation Accuracy:** 96.23%
- **Validation AUC:** ~0.995
- **Validation Precision:** ~96.1%
- **Validation Recall:** ~95.8%

### Training Insights
- The model achieved ~99% training accuracy while maintaining strong generalization.
- Early stopping was applied to prevent overfitting.
- Model weights were restored from the epoch with the highest validation accuracy.

These results indicate that the model effectively distinguishes between the four Alzheimer’s disease stages with high reliability.


 Tools & Technologies

Python

TensorFlow / Keras

NumPy

Matplotlib

Google Colab

KaggleHub

 Features

Multi-class MRI image classification

Handles class imbalance during training

Model saving and fine-tuning

Function for testing single images with visualization

📁 Project Structure
├── data/
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
├── notebooks/
│   └── Alzheimer_Classification.ipynb
├── saved_model/
├── README.md

 Author

ADIJAT Motunrayo Oyetoke
AI / Machine Learning Enthusiast

 Future Improvements

Improve recall for minority classes

Experiment with EfficientNet and MobileNet

Add Grad-CAM visualization for model interpretability

Deploy as a simple web app
