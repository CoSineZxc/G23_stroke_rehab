# Instructions
1. Place both scripts (preprocessing.py and SVM.py) along with all relevant data files into the same folder.
2. First, run preprocessing.py to preprocess the data.
3. After preprocessing, run SVM.py to perform classification.

# Workflow Overview
## Preprocessing
The following steps are applied to the raw data during preprocessing:

+ Re-referencing: Common Average Referencing (CAR) is applied, which calculates the average signal across all channels as a reference.
+ Line Noise Removal: A notch filter is used to remove power line noise at 50Hz. (Note: Most raw data has already undergone line noise removal, except for the P1 Post dataset, where an incorrect frequency was used.)
+ Bandpass Filtering: A 4th-order Butterworth zero-phase filter is applied to bandpass the data between 8-30Hz, targeting motor imagery (MI) oscillations.
+ Epoching: The data is segmented into 3-second windows, starting 2 seconds after the onset of each trial.
+ Noise Removal: In each trial, channels with voltages exceeding ±100 μV are considered noisy and are interpolated using the spline method.

## Common Spatial Pattern (CSP)
Four sub-filters are extracted using the CSP algorithm for feature extraction.

## Support Vector Machine (SVM)
A linear Support Vector Machine (SVM) is used for classification. For the training data, 4-fold cross-validation is performed to identify the best hyperparameters. Once trained, the classifier is applied to the test data.