# IMDB Review Classification

This project uses a Jupyter Notebook to classify movie reviews from the IMDB dataset as positive or negative based on the text content, using a neural network model and Word2Vec embeddings.


[Overview](#overview)

[Techniques Covered](#techniques-covered)

[Features](#features)

[Usage](#usage)

[Dependencies](#dependencies)

[Results](#results)


## OverView
The goal of this project is to classify movie reviews from the IMDB dataset into two categories: positive and negative. We use Word2Vec to represent the reviews as dense word vectors and train a neural network to classify the sentiment of the reviews. The project includes data loading, model training, and evaluation with accuracy, a classification report, and a confusion matrix.

## Techniques Covered
- Word2Vec: Word embeddings to represent text data as dense vectors.
- Neural Network: A deep learning model for binary classification (positive or negative sentiment).
- Model Evaluation: Accuracy, classification report, and confusion matrix for evaluating the model.
- Data Preprocessing: Text decoding, Word2Vec embeddings, and averaging word vectors.

## Features
- Word2Vec Embeddings: Converts text data into dense word vectors.
- Neural Network Classification: Trains a neural network to classify movie reviews as positive or negative.
- Metrics: Includes accuracy, classification report, and confusion matrix for model evaluation.
- Data Preprocessing: Reviews are preprocessed using Word2Vec embeddings to represent each review as a vector.

## Usage
- Load and preprocess the IMDB dataset: The dataset is loaded and reviews are decoded from integer-encoded values.
- Convert reviews to Word2Vec embeddings: Reviews are transformed into vector representations using Word2Vec.
- Train a neural network classification model: The neural network is trained on the vectorized reviews to classify sentiment.
- Evaluate the model: After training, the model’s performance is evaluated using accuracy, a classification report, and a confusion matrix.

## Dependencies
```
pandas
numpy
scikit-learn
tensorflow
gensim

```
## Results
- Accuracy: Overall performance of the model on test data.
- Classification Report: Provides precision, recall, and F1-score for each sentiment class (positive, negative).
- Confusion Matrix: Displays the true positive, true negative, false positive, and false negative predictions.

### Sample Output
#### Test accuracy
```
Test Accuracy: 88.35%
```
#### Classification Report
```
precision    recall  f1-score   support

        0       0.90      0.86      0.88      1250
        1       0.87      0.92      0.89      1250

    accuracy                           0.89      2500
   macro avg       0.89      0.89      0.89      2500
weighted avg       0.89      0.89      0.89      2500
```
#### Confusion Matrix
```
[[1072  178]
 [ 100 1150]]
```
