# ML-Assignment-2

## Machine Learning Assignment 2 - Bank Marketing Classification

## 1. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for predicting whether a customer will subscribe to a bank term deposit.

The project uses the Bank Marketing dataset and evaluates five different classification algorithms using multiple performance metrics.

The main objective is to compare the models and identify the best-performing model based on the evaluation results.

---

## 2. Dataset Description

### Dataset Name

Bank Marketing Dataset

### Problem Type

Binary Classification

### Target Variable

`y`

The target variable indicates whether the customer subscribed to a term deposit:

- `yes` - Customer subscribed
- `no` - Customer did not subscribe

The dataset contains customer demographic information, account-related information, and campaign-related attributes.

The Bank Marketing dataset contains 45,211 records and 16 predictor features along with the target variable.

---

## 3. GitHub Repository

GitHub Repository:

https://github.com/2025ac05097/ML-Assignment-2

The repository contains:

- Complete source code
- Trained machine learning models
- Test dataset
- Requirements file
- README documentation
- Streamlit application

---

## 4. Machine Learning Models Used

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

All five models were trained and evaluated using the same Bank Marketing classification problem.

---

## 5. Evaluation Metrics

The following five evaluation metrics were used:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

These metrics provide a comprehensive evaluation of the classification performance instead of relying only on accuracy.

---

## 6. Model Performance Comparison

The following results were obtained from the test dataset used in the project.

| ML Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.9008 | 0.8655 | 0.6186 | 0.3970 | **0.4836** | **0.4446** |
| KNN | 0.8996 | 0.8513 | 0.6298 | 0.3440 | 0.4450 | 0.4169 |
| Gaussian Naive Bayes | 0.8548 | 0.8101 | 0.4059 | **0.5198** | 0.4559 | 0.3774 |
| Random Forest | 0.8977 | **0.9225** | **0.7509** | 0.1881 | 0.3008 | 0.3427 |

---

## 7. Model-wise Observations

### 7.1 Logistic Regression

Logistic Regression achieved an accuracy of 0.9012 and an ROC-AUC of 0.9056.

It achieved a precision of 0.6445, indicating reasonable performance in identifying customers predicted as subscribers.

However, its recall was 0.3478, meaning that a significant number of actual subscribers were not identified.

The F1-Score of 0.4518 indicates moderate overall balance between precision and recall.

---

### 7.2 Decision Tree

Decision Tree achieved an accuracy of 0.9008 and an ROC-AUC of 0.8655.

It achieved a recall of 0.3970, which was higher than Logistic Regression and KNN.

Most importantly, Decision Tree achieved the highest F1-Score of 0.4836 among all five models.

It also achieved the highest MCC score of 0.4446.

Therefore, based on the F1-Score and MCC, Decision Tree provided the best overall balance between precision and recall in this experiment.

---

### 7.3 K-Nearest Neighbors (KNN)

KNN achieved an accuracy of 0.8996 and an ROC-AUC of 0.8513.

Its precision was 0.6298, showing reasonable positive-class prediction performance.

However, its recall was relatively low at 0.3440.

The resulting F1-Score was 0.4450, which was lower than Logistic Regression and Decision Tree.

Therefore, KNN provided reasonable accuracy but comparatively weaker performance in identifying actual subscribers.

---

### 7.4 Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 0.8548 and an ROC-AUC of 0.8101.

Its precision was 0.4059, which was relatively low.

However, Gaussian Naive Bayes achieved the highest recall among all five models at 0.5198.

This means it was the most successful model at identifying customers who actually subscribed to the term deposit.

The F1-Score was 0.4559, which was lower than the Decision Tree F1-Score.

Therefore, Naive Bayes may be useful when identifying as many potential subscribers as possible is more important than minimizing false-positive predictions.

---

### 7.5 Random Forest

Random Forest achieved an accuracy of 0.8977 and the highest ROC-AUC of 0.9225.

It also achieved the highest precision of 0.7509 among all five models.

However, its recall was only 0.1881, which was the lowest recall among the five models.

As a result, its F1-Score was only 0.3008.

Therefore, although Random Forest achieved excellent ROC-AUC and precision, its low recall resulted in weaker overall F1 performance for this classification task.

---

## 8. Overall Winner

### Decision Tree

Based on the evaluation results, **Decision Tree** was identified as the overall best-performing model.

It achieved:

- Highest F1-Score: **0.4836**
- Highest MCC: **0.4446**
- Accuracy: **0.9008**
- Precision: **0.6186**
- Recall: **0.3970**
- ROC-AUC: **0.8655**

The Decision Tree provided the best overall balance between Precision and Recall according to the F1-Score criterion used for model ranking.

Although Random Forest achieved the highest ROC-AUC and Precision, its low Recall resulted in a much lower F1-Score.

Gaussian Naive Bayes achieved the highest Recall, but its lower Precision resulted in a lower F1-Score than Decision Tree.

Therefore, **Decision Tree is selected as the overall winner based on the highest F1-Score.**

---

## 9. Streamlit Application

An interactive Streamlit application was developed using `app.py`.

The application provides the following features:

- Select a machine learning model
- Upload the test dataset
- Preview the uploaded dataset
- Display dataset shape
- Generate predictions
- Display model performance metrics
- Display the confusion matrix
- Display the classification report
- Compare the performance of all five machine learning models

The application loads the trained model files stored in the `model` directory.

---

## 10. Project Structure

```text
ML-Assignment-2/
│
├── app.py
├── README.md
├── requirements.txt
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

---

## 11. Installation

Install the required Python packages using:

```bash
py -m pip install -r requirements.txt
```

---

## 12. Running the Application

Start the Streamlit application using:

```bash
py -m streamlit run app.py
```

The application will open in the browser.

---

## 13. Using the Application

1. Start the Streamlit application.
2. Select a machine learning model from the sidebar.
3. Upload `test_data.csv`.
4. Review the dataset preview.
5. View the prediction results.
6. Review the performance metrics.
7. View the confusion matrix.
8. View the classification report.
9. Compare all five models.

---

## 14. Trained Models

The following trained model files are included in the `model` directory:

- `logistic_regression.pkl`
- `decision_tree.pkl`
- `knn.pkl`
- `naive_bayes.pkl`
- `random_forest.pkl`

These saved models are loaded by the Streamlit application during execution.

---

## 15. Conclusion

This project demonstrates an end-to-end machine learning classification workflow using the Bank Marketing dataset.

Five classification models were implemented and evaluated using Accuracy, ROC-AUC, Precision, Recall, F1-Score, and MCC.

The experimental results show that:

- **Decision Tree** achieved the highest F1-Score and was selected as the overall winner.
- **Gaussian Naive Bayes** achieved the highest Recall.
- **Random Forest** achieved the highest ROC-AUC and Precision.
- **Logistic Regression** provided strong and balanced overall performance.
- **KNN** achieved reasonable accuracy but comparatively lower recall and F1-Score.

The Streamlit application provides an interactive interface for testing the trained models and visualizing their classification performance.
