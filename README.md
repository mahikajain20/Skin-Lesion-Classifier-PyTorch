# **Skin Lesion Classification using PyTorch and Stanford DDI Dataset**

***Final Project Lighthouse Labs (Data Science Bootcamp)**

## Overview

This project aims to develop a deep learning model for classifying skin lesions using the Stanford Diverse Dermatology Images (DDI) dataset. 

We'll be using PyTorch to build and train a Convolutional Neural Network (CNN) for this binary classification task.

The goal is to accurately distinguish between two classes of skin lesions, which can aid in early detection and diagnosis of skin conditions. This project will explore various aspects of deep learning, including data preparation, model architecture design, and hyperparameter tuning.

- **Tech Stack:**
  - PyTorch and torchvision for model development.
  - Scikit-learn for evaluation metrics.

- **Key objectives:**

1. Preprocess and analyze the DDI dataset
2. Design and implement a CNN using PyTorch
3. Train and evaluate the model using various techniques
4. Compare different model architectures and hyperparameters
5. Analyze results 
6. Deploy the model and provide recommendations for future work

## **Table of Contents**
1. [Project Motivation](#project-motivation)
2. [Dataset](#data-description)
3. [Model Architecture](#model-architecture)
4. [Training and Evaluation](#training-and-evaluation)
5. [Inference](#inference)
6. [Troubleshooting](#troubleshooting)
7. [Future Work](#future-work)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Contributing](#contributing)
11. [Acknowledgements](#acknowledgements)

## **Project Motivation**
An estimated 3 billion people worldwide lack access to adequate medical care for skin diseases. This project addresses this critical issue by developing a model that can assist in the early detection of skin cancer, helping reduce wait times and supporting non-specialist physicians in diagnosing skin diseases.

##  Data Description

The Stanford Diverse Dermatology Images (DDI) dataset is used for this project. It consists of diverse skin lesion images categorized into two classes. The dataset aims to provide a more inclusive representation of skin conditions across different skin types and ethnicities.

Dataset characteristics:
- **Name:** Diverse Dermatology Images (DDI)

- **Description:** The dataset contains images of skin lesions with diverse skin tones and expert-labeled data, which ensures fair performance across all skin tones.

- Source: Stanford University
- Number of classes: 2
- Image format: PNG
- Image dimensions: [224x224 pixels]
- Total number of images: 656
- Class distribution: 485 Benign, 171 Malignant 

The dataset is split into training, validation, and test sets to ensure proper model evaluation and prevent overfitting.

***Reference:*** Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set. Roxana Daneshjou, Kailas Vodrahalli, Weixin Liang, Roberto A Novoa, Melissa Jenkins, Veronica Rotemberg, Justin Ko, Susan M Swetter, Elizabeth E Bailey, Olivier Gevaert, Pritam Mukherjee, Michelle Phung, Kiana Yekrang, Bradley Fong, Rachna Sahasrabudhe, James Zou, Albert Chiou. Science Advances (2022)


## Model Architecture

### Model 1 : Custom ResNet50 

The model leverages the concept of residual learning to ease the training of such deep networks. The main innovation of ResNet is the introduction of "skip connections" which allow the gradient to be directly backpropagated to earlier layers.

### Model 2: Custom DenseNet121

DenseNet connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections—one between each layer and its subsequent layer—DenseNet has L(L+1)/2 direct connections.

### Loss function and optimizer

Loss function; Used a weighted loss function for models E and F. 

Optimizer: Adam (Adaptive Moment Estimation) is an efficient and widely used optimization algorithm that computes adaptive learning rates for each parameter.

ReduceLROnPlateau: This learning rate scheduler reduces the learning rate when a metric has stopped improving. It is useful for training models where the performance plateaus at certain learning rates. 

## Training and Evaluation 

- **Data Augmentation:** For the training dataset, the transformations include data augmentation techniques such as random resizing and cropping, horizontal flipping, rotation, and color jittering. 

For the validation and test datasets, the transformations are simpler; No data augmentation is applied to these datasets as they serve as a 'ground truth' for model evaluation.

- **Class Imbalance:** Addressed using a weighted loss function. This allowed the model to give more importance to underrepresented classes during training.

- **Evaluation Metrics:** The models were evaluated using accuracy, precision, recall, loss, confusion matrices and ROC-AUC. These metrics were calculated for each class, ensuring a comprehensive understanding of model performance.

### Results 

| Model | Architecture | Dropout Rate | Training Loss | Training Accuracy | Training AUC | Validation Loss | Validation Accuracy | Validation AUC |
|-------|--------------|--------------|---------------|-------------------|--------------|-----------------|---------------------|----------------|
| A     | ResNet       | 0.1          | 0.4598        | 0.8065            | 0.6720       | 0.5306         | 0.7862             | 0.6455         |
| B     | DenseNet     | 0.1          | 0.8760        | 0.5087           | 0.5098       | 0.6443          | 0.6564              | 0.5675         |
| C     | ResNet       | 0.3          | 0.2476       | 0.900            | 0.8550       | 0.7290          | 0.7862              | 0.6455         |
| D     | DenseNet     | 0.3          | 0.5028        | 0.8196            | 0.7580       | 0.5131         | 0.7826              | 0.6169         |
| E     | ResNet  WL     | 0.3          | 0.6198        | 0.6978            | 0.6754       | 1.094          | 0.7710              | 0.6448         |
| F     | DenseNet  WL   | 0.3          | 0.1784     | 0.9391            | 0.9427       | 0.7966          | 0.7826            | 0.6932         |

- **Best Performing Model**: Model F 

Model F has the highest training accuracy (0.9391) and also the highest area under the ROC curve (AUC) for the training set (0.9428), indicating that it performs very well on the training data. It also has the highest validation AUC (0.6933), which suggests that it generalizes well to unseen data. However, its validation accuracy is the same as Models A, C, and D (0.7863).


- **Ensemble Model**: All the 6 models were trained and then their predictions were combined using Ensemble voting to make a final prediction. This accuracy was 80% and AUC was 0.64, suggesting high accuracy but can improve discernment between classes.

**. Impact of Dropout Rate**

- Low Dropout : Models A and B have low dropout rates. Model A has a relatively high training accuracy (0.8065) and validation accuracy (0.7862), suggesting that a low dropout rate works well for this model. However, Model B has a lower training accuracy (0.5087) and validation accuracy (0.6564), indicating that a low dropout rate might not be sufficient for this model.


- Moderate Dropout : Models C, D, E, and F have moderate dropout rates. These models show varying degrees of performance, with Model F having the highest validation accuracy (0.7826) and AUC (0.6932), suggesting that a moderate dropout rate might be beneficial for this model.

**. Impact of Class weights on Loss Function**

- Class Imbalance : Models E and F were trained with class weights to account for class imbalance. Model F, which has the highest validation AUC (0.6932), might have found a better balance of class weights, but further investigation is needed.

## Inference

The ***inference pipeline*** involves feeding an image into the trained model, which then outputs a classification label (malignant or benign) along with the probability scores for each category.

**Deployment:** The trained model was deployed using Flask, creating a web application where users can upload images of skin lesions for instant classification.

**User Interface:** The Flask app allows users to interact with the model by uploading images and receiving predictions, complete with probability distributions for each class.


## Troubleshooting

Monitoring Training and Validation Metrics:

- Loss and Accuracy: I regularly monitored both training and validation loss and accuracy during each epoch to detect signs of overfitting or underfitting.

- Graphs: I plotted the training and validation loss and accuracy curves to visually inspect the model's performance and trends over time.

Adjusting Dropout Rate:

- Overfitting: When the model exhibited high training accuracy but low validation accuracy, I increased the dropout rate to add more regularization and prevent overfitting.

- Underfitting: If both training and validation accuracies were low, I reduced the dropout rate to allow the model to learn more effectively from the data.

Modifying Layers:

- Complexity: If the model was overfitting with a high number of convolutional layers, I reduced the number of layers to simplify the model.

- Capacity: Conversely, if the model was underfitting with fewer convolutional layers, I increased the number of layers to enhance the model's capacity to learn complex features.

Hyperparameter Tuning:

- Learning Rate: I experimented with different learning rates to find the optimal value that allowed the model to converge effectively without oscillating or diverging.

- Batch Size: I adjusted the batch size to observe its impact on the model's learning stability and convergence speed.

Early Stopping and Learning Rate Scheduling:

- Early Stopping: I implemented early stopping to halt training when the validation loss stopped improving for a set number of epochs, thereby preventing overfitting.

- Learning Rate Scheduler: I used a learning rate scheduler to reduce the learning rate when the validation loss plateaued, allowing the model to fine-tune its learning process.

By following these troubleshooting procedures, I iteratively refined the model configurations to achieve the best balance between training and validation performance.


## Future Work 


To further enhance this project and potentially improve its real-world applicability, several avenues for future work can be explored:

- Multi-class Classification: Extend the model to classify multiple types of skin lesions beyond the current binary classification, providing more detailed diagnostic information.

- Explainable AI: Implement techniques like Grad-CAM or SHAP values to visualize and interpret which parts of the images are most influential in the model's decisions, enhancing trust and interpretability.

- External Validation: Test the model on external datasets to assess its generalization capabilities across different populations and image acquisition settings.

- Integration with Clinical Data: Combine image data with patient metadata (age, skin type, medical history) to create a more comprehensive diagnostic tool.
Mobile Application: Develop a mobile app that allows users to take photos of skin lesions and receive instant preliminary assessments, promoting early detection and screening.

- Continual Learning: Implement a system for continual model updating as new data becomes available, ensuring the model stays current with the latest examples and potential variations in skin lesions.

- Federated Learning: Explore federated learning techniques to train models across multiple healthcare institutions without sharing sensitive patient data, addressing privacy concerns and potentially increasing the diversity of the training data.

- Comparative Study: Conduct a comprehensive study comparing the performance of the developed AI model against dermatologists of varying experience levels to benchmark its practical utility.

- Using diffuse learning models and LLM integration to help out with patient records and improving accuracy.

By pursuing these future directions, the project can evolve into a more robust, clinically relevant tool for skin lesion classification, potentially contributing to improved early detection and diagnosis of skin conditions. 

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/https://github.com/mahikajain20/Skin-Lesion-Classifier-PyTorch.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## **Usage**
1. Run the Flask app:
   ```bash
   python app.py
   ```
2. Upload an image through the web interface to classify it.

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request.

## **Acknowledgements**

- Lighthouse Labs for this opportunity to learn from the Data Science Bootcamp! 

- IDRF for giving me the Women in Tech scholarship to pursue this program (valued at $14,000) !

- Stanford University for providing the DDI dataset.

- PyTorch and scikit-learn communities for their tools and resources.