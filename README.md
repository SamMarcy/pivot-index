# CVIP-VISION-CHALLENGE Submission: CAVE-NET

## Overview

This repository contains our solution, **CAVE-NET**, for the CVIP-VISION-CHALLENGE, focusing on multi-class abnormality classification in medical imaging. CAVE-NET addresses class imbalance through data augmentation and applies an ensemble of deep learning models for robust classification.

## Repository Structure

1. **Data Augmentation** (`augmentation.ipynb`)  
   The initial dataset presented significant class imbalances, impacting model accuracy. The augmentation notebook addresses this imbalance by creating a balanced dataset through data augmentation. **Run this notebook first** to prepare the dataset.

2. **CAVE-NET Model** (`encoderdecoder.ipynb`)  
   This notebook contains the core model implementation of CAVE-NET, which follows these steps:
   - **Encoder-Decoder Architecture**: Generates **latent spaces** and **reconstructed images** from the augmented dataset.
   - **CBAM Model**: A Convolutional Block Attention Module model trained on both original and reconstructed images to enhance feature extraction and improve classification.
   - **Deep Neural Network (DNN)**: Trained on latent spaces produced by the encoder, it leverages high-level image representations.
   - **Ensemble Model (SVM, KNN, XGB, Random Forest)**: A set of classifiers trained on the latent space for enhanced model diversity and robustness.
   - **Final Ensemble**: This soft-voting ensemble combines predictions from the CBAM, DNN, and ensemble models for the final output.

## Instructions

1. **Run Data Augmentation**:  
   ```bash
   jupyter notebook augmentation.ipynb
   ```

2. **Run CAVE-NET Model**:  
   This notebook handles latent space extraction, model training, and final prediction generation.
   ```bash
   jupyter notebook encoderdecoder.ipynb
   ```

## Results

CAVE-NET achieved **91.2% accuracy** on both training and validation datasets, demonstrating its effectiveness in handling class imbalance and improving classification performance through a multi-faceted ensemble approach.

## Acknowledgements

We extend our gratitude to the CVIP-VISION-CHALLENGE organizers for providing a meaningful platform to explore innovative solutions in medical imaging.
