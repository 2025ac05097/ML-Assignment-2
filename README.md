# ML-Assignment-2

## Machine Learning Assignment 2 - Bank Marketing Classification

This project implements a machine learning classification solution for the Bank Marketing dataset. The objective is to predict whether a customer will subscribe to a term deposit based on customer and campaign-related attributes.

## Machine Learning Models

The following five classification models are implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

## Evaluation Metrics

The models are evaluated using the following performance metrics:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

## Model Comparison

All five machine learning models are compared using the above performance metrics.

The models are ranked based on the highest F1-Score, as F1-Score provides a balanced measure of Precision and Recall.

Based on the evaluation results, the Decision Tree model achieved the highest F1-Score of 0.4836 and was identified as the overall best-performing model among the five models evaluated.

## Streamlit Application

An interactive Streamlit application is provided through `app.py`.

The application allows the user to:

- Select a machine learning model
- Upload the test dataset
- View model performance metrics
- View the confusion matrix
- View the classification report
- Compare the performance of all five machine learning models

## Installation

Install the required Python packages using:

```bash
py -m pip install -r requirements.txt

Add:

```markdown
## Running the Application

Start the Streamlit application using:

```bash
py -m streamlit run app.py
