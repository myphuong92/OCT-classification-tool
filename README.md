# Optical Coherence Tomography (OCT) Scan Classification using Deep Neural Network
This repository contains the code for the visualization tool of the final thesis project of the HTTT2020 class at the University of Information Technology, Ho Chi Minh City.

## Project Description
The project aims to classify Optical Coherence Tomography (OCT) scans using deep neural networks. It provides a user interface for selecting different models and utilizes ensemble models based on Fuzzy Ranking to classify uploaded images.

## Models
- Inception-V3
- ResNet 50
- DenseNet 201
- Efficientnet-B3

## Fuzzy Rank-based Fusion using Ensemble Functions for Deep Learning Models
In addition to the individual models mentioned above, we have also utilized ensemble models to enhance the classification performance. These models combine predictions from multiple individual models using ensemble functions. The weights assigned to each model are based on their performance and reliability. By leveraging the strengths of different models, this fusion technique improves the overall accuracy and robustness of the classification system.
  
## Usage
To utilize the project, follow the steps below:

1. Select a model from the dropdown list. You can choose to make predictions using a single independent model or use ensemble functions to combine multiple models.
2. Upload an image for classification.
3. Click on the "Submit" button.
The classification results will be displayed below.

## Installation
No installation is required for this project. 

**Note:** The tool has been deployed and can be accessed at domain.com