# ML-Assignment-2

## Machine Learning Assignment 2 - Bank Marketing Classification

This project implements a machine learning classification solution for the Bank Marketing dataset. The objective is to predict whether a customer will subscribe to a term deposit based on customer and campaign-related attributes.

## Machine Learning Models

The following five classification models are implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest

## Evaluation Metrics

The models are evaluated using the following performance metrics:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

## Streamlit Application

An interactive Streamlit application is provided through `app.py`.

The application allows the user to:

- Select a machine learning model
- Upload the test dataset
- View model performance metrics
- View the confusion matrix
- View the classification report
- Compare the performance of all five machine learning models

## Project Structure

```text
ML-Assignment-2/
│
├── app.py
├── README.md
├── requirements.txt
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
