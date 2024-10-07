# Riemannian Geometry-Based EEG Classification

This repository contains two Python scripts for the classification of EEG signals using a Riemannian geometry-based approach. The workflow includes data preprocessing followed by classification using a Support Vector Machine (SVM). The goal of the classification task is to distinguish between different motor imagery (MI) tasks using EEG signals.

## Workflow Overview

### Preprocessing
- **Re-referencing**: Common Average Referencing (CAR) is applied to calculate the average signal across all EEG channels and use it as a reference.
- **Line Noise Removal**: A notch filter is applied to remove power line noise (50Hz) from the data.
- **Bandpass Filtering**: A 4th-order Butterworth zero-phase filter is used to bandpass the data between specific frequency ranges (e.g., 8-30Hz) to isolate motor imagery oscillations.
- **Epoching**: The data is segmented into 3-second windows, starting 2 seconds after the onset of each trial.
- **Noise Removal**: Channels with voltages exceeding ±100 µV are considered noisy and are interpolated using the spline method.

### Feature Extraction using Riemannian Geometry
- **Covariance Matrices**: The covariance matrices of the EEG signals are computed as features, representing the spatial relationships between different EEG channels.
- **Tangent Space Projection**: To make these covariance matrices suitable for classification, they are projected onto a tangent space at the Riemannian mean, providing a linear representation of the original non-linear data.

### Classification using SVM
- **Support Vector Machine (SVM)**: A linear SVM classifier is trained using the features obtained from the tangent space projection. Cross-validation is used on the training data to determine the best hyperparameters, and the trained model is used to classify the test data.

## How to Use
1. Place both scripts (`preprocessing.py` and `RMN-SVM.py`) along with the relevant data files in the same directory.
2. Run `preprocessing.py` to preprocess the EEG data.
3. After preprocessing, run `RMN-SVM.py` to perform the classification.

## Requirements
- Python 3.x
- NumPy
- SciPy
- scikit-learn
- MNE
- pyRiemann

## Results
The Riemannian geometry-based approach aims to better capture the non-linear relationships in EEG data, which leads to improved classification accuracy compared to traditional methods, especially for motor imagery tasks.
