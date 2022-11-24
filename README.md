# CDT-API-NETWORK

This repository contains the dataset and model code used in  **Attentive Pairwise Interaction Network for AI-assisted Clock Drawing Test Assessment of Early Visuospatial Deficits**.
In `clock_shulman.zip`, internal folder is splited by shulman clock score.

# Introduction
Dementia is a debilitating neurological condition which impairs the cognitive function and the ability to take care of oneself. The Clock Drawing Test (CDT) is a simple and well-known test for detecting dementia. While clear abnormalities in CDT indicate high risk of dementia, discriminating between normal versus borderline cases requires years of clinical experience. Misclassifying mild abnormal as normal will delay the chance to investigate for potential reversible causes or slow down the progression.
To help address this issue, we propose an automatic CDT scoring system that adopts Attention Pairwise Interaction Network (API-Net), a fine-grained deep learning model that is designed to detect subtle differences in images.
Inspired by how humans often learn to recognize different objects by looking at two images side-by-side, the API-Net learns to detect the subtle differences by comparing image pairs contrastively.

# How to use
** The provided training script is just an example for creating training pipeline.

Create Conda Environment:
` conda env create -f environment.yml `

Then run:
` py train.py `

# Model performance
| Model | Accuracy | F1-Score | Precision | Recall |
| :-----: | :---: | :---: | :---: | :---: |
| ResNet-152  | 0.7668±0.0074  | 0.7581±0.0079   | 0.7654±0.0080   | 0.7668±0.0074   |
| API-Net (ResNet-152)  | 0.7802±0.0030 | 0.7653±0.0084 | 0.7727±0.0057 | 0.7743±0.0054 |
| API-Net (ResNet-121) with gradual unfreezing  | 0.7892±0.0104  | 0.7964±0.0120  | 0.7871±0.0108  | 0.7892±0.0104  |

# Contact:
Please feel free to contact raksit.r@hotmail.com or chaipat.c@chula.ac.th, if you have any questions.