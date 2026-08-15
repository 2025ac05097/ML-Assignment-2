# ML-Assignment-2

## Machine Learning Assignment 2 - Bank Marketing Classification

This project implements a machine learning classification solution for the Bank Marketing dataset. The objective is to predict whether a customer will subscribe to a term deposit based on customer and campaign-related attributes.

## 1. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for predicting whether a customer will subscribe to a bank term deposit.

The project evaluates five classification algorithms using multiple performance metrics. The aim is to identify the model that provides the best overall balance between precision and recall.

---

## 2. Machine Learning Models Used

The following five classification models are implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

All five models are evaluated using the same test dataset to provide a consistent comparison.

---

## 3. Evaluation Metrics

The models are evaluated using the following performance metrics:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

These metrics provide a more complete evaluation of classification performance, particularly for the positive class.

---

## 4. Model Performance Comparison

The following table shows the performance of all five machine learning models:

| Rank | Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score | MCC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Decision Tree | 0.9008 | 0.8655 | 0.6186 | 0.3970 | **0.4836** | 0.4446 |
| 2 | Gaussian Naive Bayes | 0.8548 | 0.8101 | 0.4059 | **0.5198** | 0.4559 | 0.3774 |
| 3 | Logistic Regression | **0.9012** | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| 4 | KNN | 0.8996 | 0.8513 | 0.6298 | 0.3440 | 0.4450 | 0.4169 |
| 5 | Random Forest | 0.8977 | **0.9225** | **0.7509** | 0.1881 | 0.3008 | 0.3427 |

The models are ranked based on F1-Score because F1-Score provides a balanced measure of Precision and Recall.

---

## 5. Model-wise Observations

### 5.1 Logistic Regression

Logistic Regression achieved an accuracy of 0.9012 and an ROC-AUC of 0.9056.

Its precision was 0.6445, while its recall was 0.3478. The model provides a strong baseline classification performance but does not identify as many positive cases as Gaussian Naive Bayes.

The F1-Score of Logistic Regression was 0.4518.

---

### 5.2 Decision Tree

Decision Tree achieved an accuracy of 0.9008 and an ROC-AUC of 0.8655.

Its precision was 0.6186 and recall was 0.3970. The model achieved the highest F1-Score among all five models:

**F1-Score = 0.4836**

Therefore, based on the selected F1-Score criterion, Decision Tree was identified as the overall best-performing model.

---

### 5.3 K-Nearest Neighbors (KNN)

KNN achieved an accuracy of 0.8996 and an ROC-AUC of 0.8513.

Its precision was 0.6298 and recall was 0.3440.

The resulting F1-Score was 0.4450, which was lower than the F1-Scores of Decision Tree, Gaussian Naive Bayes, and Logistic Regression.

---

### 5.4 Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 0.8548 and an ROC-AUC of 0.8101.

Its precision was 0.4059.

However, Gaussian Naive Bayes achieved the highest recall among all five models:

**Recall = 0.5198**

This means that it identified the largest proportion of actual positive cases.

Its F1-Score was 0.4559.

---

### 5.5 Random Forest

Random Forest achieved an accuracy of 0.8977 and an ROC-AUC of 0.9225.

It achieved the highest precision among all five models:

**Precision = 0.7509**

It also achieved the highest ROC-AUC:

**ROC-AUC = 0.9225**

However, its recall was only 0.1881, resulting in an F1-Score of 0.3008.

Therefore, although Random Forest performed best on ROC-AUC and Precision, its lower recall resulted in a lower F1-Score.

---

## 6. Overall Model Selection

Based on the F1-Score comparison, the **Decision Tree** model was selected as the overall best-performing model.

### Decision Tree

- Accuracy: 0.9008
- ROC-AUC: 0.8655
- Precision: 0.6186
- Recall: 0.3970
- F1-Score: **0.4836**
- MCC: 0.4446

Decision Tree achieved the highest F1-Score of **0.4836** among the five evaluated models.

However, different models performed best on individual metrics:

- **Highest Accuracy:** Logistic Regression - 0.9012
- **Highest ROC-AUC:** Random Forest - 0.9225
- **Highest Precision:** Random Forest - 0.7509
- **Highest Recall:** Gaussian Naive Bayes - 0.5198
- **Highest F1-Score:** Decision Tree - 0.4836
- **Highest MCC:** Decision Tree - 0.4446

Therefore, Decision Tree is considered the overall best model based on the F1-Score criterion used in this project.

---

## 7. Streamlit Application

An interactive Streamlit application is provided through `app.py`.

The application allows the user to:

- Select a machine learning model
- Upload the test dataset
- View model performance metrics
- View the confusion matrix
- View the classification report
- Compare the performance of all five machine learning models

The application uses the saved trained model files stored in the `model` directory.

---

## 8. Application Features

The Streamlit application provides the following functionality:

### Model Selection

The user can select one of the five trained machine learning models:

- Logistic Regression
- Decision Tree
- KNN
- Gaussian Naive Bayes
- Random Forest

### Test Dataset Upload

The application provides an option to upload the test dataset in CSV format.

### Performance Summary

After prediction, the application displays:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-Score
- MCC

### Confusion Matrix

The application displays the confusion matrix for the selected model to show the classification results in terms of:

- True Negatives
- False Positives
- False Negatives
- True Positives

### Classification Report

The application also provides the classification report for the selected model.

### Model Comparison

The application provides a comparison of all five machine learning models based on the evaluation metrics.

---

## 9. Installation

Install the required Python packages using:

```bash
py -m pip install -r requirements.txt
```

---

## 10. Running the Application

Start the Streamlit application using:

```bash
py -m streamlit run app.py
```

The application will open in the browser and provide an interactive interface for model selection, test dataset upload, prediction, evaluation, and model comparison.

---

## 11. Project Structure

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
```

---

## 12. Conclusion

Five machine learning classification models were implemented and evaluated for the Bank Marketing classification task.

The models were compared using Accuracy, ROC-AUC, Precision, Recall, F1-Score, and Matthews Correlation Coefficient.

The **Decision Tree** achieved the highest F1-Score of **0.4836** and was therefore selected as the overall best-performing model based on the F1-Score criterion.

The results also show that different models have different strengths. Random Forest achieved the highest ROC-AUC and Precision, while Gaussian Naive Bayes achieved the highest Recall.

The Streamlit application provides an interactive way to select models, upload test data, view predictions and evaluation metrics, visualize the confusion matrix, view the classification report, and compare all five models.
