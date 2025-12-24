# LConvNet-SCA-EEG
# Scaled Custom Attention for EEG Classification

This repository provides the implementation of **Scaled Custom Attention (SCA)**
integrated into an LConvNet architecture for EEG-based epilepsy classification.

## Dataset
- Temple University Hospital (TUH) EEG Epilepsy Corpus
- Dataset is not redistributed due to license restrictions

## Preprocessing
- Average referencing
- Bandpass filtering (1–45 Hz)
- Resampling to 128 Hz
- Cropping to 200 seconds
- Epoching: 2 s windows with 50% overlap
- PCA dimensionality reduction to 25 components

## Model
- CNN-based spatial feature extraction
- TimeDistributed embedding
- Scaled Custom Attention (SCA)
- LSTM temporal modeling
- Binary classification output

## How to Run
```bash
python train.py

