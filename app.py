import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Classification",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Classification")
st.write(
    "Machine Learning Assignment 2 - Bank Marketing Classification"
)

st.markdown("---")


# ============================================================
# LOAD ALL 5 MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            "model/logistic_regression.pkl"
        ),

        "Decision Tree": joblib.load(
            "model/decision_tree.pkl"
        ),

        "KNN": joblib.load(
            "model/knn.pkl"
        ),

        "Gaussian Naive Bayes": joblib.load(
            "model/naive_bayes.pkl"
        ),

        "Random Forest": joblib.load(
            "model/random_forest.pkl"
        )
    }

    return models


try:
    models = load_models()

except Exception as e:

    st.error("Unable to load the trained models.")
    st.exception(e)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Select Machine Learning Model:",
    list(models.keys())
)

selected_model = models[selected_model_name]


# ============================================================
# CSV UPLOAD
# ============================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


# ============================================================
# PROCESS DATA
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(uploaded_file)

    except Exception as e:

        st.error("Unable to read the uploaded CSV file.")
        st.exception(e)
        st.stop()


    st.success("CSV file uploaded successfully!")


    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.write(
        f"**Dataset Shape:** "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )


    # ========================================================
    # CHECK TARGET COLUMN
    # ========================================================

    if "y" not in df.columns:

        st.error(
            "The uploaded CSV must contain the target column 'y'."
        )

        st.stop()


    # ========================================================
    # PREPARE FEATURES AND TARGET
    # ========================================================

    X = df.drop(columns=["y"])

    y_raw = df["y"]


    # ========================================================
    # CONVERT TARGET
    # ========================================================

    if pd.api.types.is_numeric_dtype(y_raw):

        # Target already numeric
        y = pd.to_numeric(
            y_raw,
            errors="coerce"
        )

    else:

        # Target is text
        y = (
            y_raw
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "no": 0,
                "yes": 1
            })
        )


    # ========================================================
    # VALIDATE TARGET
    # ========================================================

    if y.isna().any():

        st.error(
            "Target column 'y' contains missing or invalid values."
        )

        st.write(
            "**Values found in y:**",
            y_raw.unique()
        )

        st.write(
            "**Number of invalid/missing target values:**",
            int(y.isna().sum())
        )

        st.stop()


    # Convert target to integer
    y = y.astype(int)


    # ========================================================
    # VALIDATE TARGET CLASSES
    # ========================================================

    invalid_classes = set(y.unique()) - {0, 1}

    if invalid_classes:

        st.error(
            f"Invalid target classes found: {invalid_classes}. "
            "Expected only 0 and 1."
        )

        st.stop()


    # ========================================================
    # TARGET INFORMATION
    # ========================================================

    with st.expander("Target Variable Information"):

        st.write(
            "**Original target values:**",
            y_raw.unique()
        )

        st.write(
            "**Encoded target values:**",
            y.unique()
        )

        target_distribution = pd.DataFrame({
            "Target": [
                "No (0)",
                "Yes (1)"
            ],

            "Count": [
                int((y == 0).sum()),
                int((y == 1).sum())
            ]
        })

        st.write("**Target distribution:**")

        st.dataframe(
            target_distribution,
            width="stretch"
        )


    # ========================================================
    # RUN PREDICTION
    # ========================================================

    if st.button(
        "Run Prediction",
        type="primary"
    ):

        with st.spinner(
            f"Running {selected_model_name}..."
        ):

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            try:

                y_pred = selected_model.predict(X)

            except Exception as e:

                st.error(
                    f"Prediction failed for {selected_model_name}."
                )

                st.exception(e)
                st.stop()


            # ------------------------------------------------
            # PREDICTION PROBABILITY
            # ------------------------------------------------

            try:

                if hasattr(
                    selected_model,
                    "predict_proba"
                ):

                    y_prob = (
                        selected_model
                        .predict_proba(X)[:, 1]
                    )

                else:

                    y_prob = None

            except Exception:

                y_prob = None


        # ====================================================
        # METRICS
        # ====================================================

        try:

            accuracy = accuracy_score(
                y,
                y_pred
            )

            precision = precision_score(
                y,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y,
                y_pred,
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y,
                y_pred
            )


            if y_prob is not None:

                roc_auc = roc_auc_score(
                    y,
                    y_prob
                )

            else:

                roc_auc = np.nan


        except Exception as e:

            st.error(
                "Unable to calculate evaluation metrics."
            )

            st.exception(e)
            st.stop()


        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown("---")

        st.header(
            f"2. {selected_model_name} Performance"
        )


        # ====================================================
        # METRIC CARDS
        # ====================================================

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

            st.metric(
                "Precision",
                f"{precision:.4f}"
            )


        with col2:

            if not np.isnan(roc_auc):

                st.metric(
                    "ROC-AUC",
                    f"{roc_auc:.4f}"
                )

            else:

                st.metric(
                    "ROC-AUC",
                    "N/A"
                )

            st.metric(
                "Recall",
                f"{recall:.4f}"
            )


        with col3:

            st.metric(
                "F1-Score",
                f"{f1:.4f}"
            )

            st.metric(
                "MCC",
                f"{mcc:.4f}"
            )


        # ====================================================
        # PERFORMANCE SUMMARY
        # ====================================================

        st.subheader("Performance Summary")

        performance_df = pd.DataFrame({

            "Metric": [
                "Accuracy",
                "ROC-AUC",
                "Precision",
                "Recall",
                "F1-Score",
                "MCC"
            ],

            "Score": [
                accuracy,
                roc_auc,
                precision,
                recall,
                f1,
                mcc
            ]
        })


        st.dataframe(
            performance_df.round(4),
            width="stretch"
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y,
            y_pred,
            labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual No",
                "Actual Yes"
            ],
            columns=[
                "Predicted No",
                "Predicted Yes"
            ]
        )


        # ----------------------------------------------------
        # Confusion Matrix Figure
        # ----------------------------------------------------

        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No", "Yes"]
        ).plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            colorbar=False
        )

        ax.set_title(
            f"Confusion Matrix - {selected_model_name}"
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        # Optional numeric values
        with st.expander("View Confusion Matrix Values"):
            st.dataframe(
                cm_df,
                width="stretch"
            )


        # ====================================================
        # CONFUSION MATRIX VALUES
        # ====================================================

        tn, fp, fn, tp = cm.ravel()

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "True Negatives",
                int(tn)
            )


        with col2:

            st.metric(
                "False Positives",
                int(fp)
            )


        with col3:

            st.metric(
                "False Negatives",
                int(fn)
            )


        with col4:

            st.metric(
                "True Positives",
                int(tp)
            )


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        st.subheader("Classification Report")

        report = classification_report(
            y,
            y_pred,
            target_names=[
                "No",
                "Yes"
            ],
            output_dict=True,
            zero_division=0
        )


        report_df = pd.DataFrame(
            report
        ).transpose()


        st.dataframe(
            report_df.round(4),
            width="stretch"
        )


        # ====================================================
        # PREDICTION SUMMARY
        # ====================================================

        st.subheader("Prediction Summary")

        prediction_counts = pd.Series(
            y_pred
        ).value_counts()


        prediction_summary = pd.DataFrame({

            "Prediction": [
                "No",
                "Yes"
            ],

            "Count": [

                int(
                    prediction_counts.get(
                        0,
                        0
                    )
                ),

                int(
                    prediction_counts.get(
                        1,
                        0
                    )
                )
            ]
        })


        st.dataframe(
            prediction_summary,
            width="stretch"
        )


        # ====================================================
        # PREDICTION RESULTS
        # ====================================================

        st.subheader("Prediction Results")

        result_df = X.copy()

        result_df["Actual"] = y.values

        result_df["Predicted"] = y_pred


        if y_prob is not None:

            result_df["Probability_Yes"] = y_prob


        st.dataframe(
            result_df.head(100),
            width="stretch"
        )


        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        csv_output = result_df.to_csv(
            index=False
        )


        st.download_button(
            label="Download Prediction Results",

            data=csv_output,

            file_name=(
                selected_model_name
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
                + "_predictions.csv"
            ),

            mime="text/csv"
        )


        # ====================================================
        # MODEL COMPARISON
        # ====================================================

        st.markdown("---")
        st.header("3. Comparison of All Five Models")

        comparison_results = []

        for model_name, model in models.items():

            try:
                model_pred = model.predict(X)

                if hasattr(model, "predict_proba"):
                    model_prob = model.predict_proba(X)[:, 1]
                    model_auc = roc_auc_score(y, model_prob)
                else:
                    model_auc = np.nan

                comparison_results.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(y, model_pred),
                    "ROC-AUC": model_auc,
                    "Precision": precision_score(
                        y, model_pred, zero_division=0
                    ),
                    "Recall": recall_score(
                        y, model_pred, zero_division=0
                    ),
                    "F1-Score": f1_score(
                        y, model_pred, zero_division=0
                    ),
                    "MCC": matthews_corrcoef(
                        y, model_pred
                    )
                })

            except Exception as e:
                st.warning(
                    f"Could not evaluate {model_name}: {e}"
                )

        comparison_df = pd.DataFrame(comparison_results)

        if not comparison_df.empty:

            best_model = comparison_df.loc[
                comparison_df["F1-Score"].idxmax(),
                "Model"
            ]

            comparison_df = comparison_df.sort_values(
                by="F1-Score",
                ascending=False
            ).reset_index(drop=True)

            comparison_df.insert(
                0,
                "Rank",
                range(1, len(comparison_df) + 1)
            )

            st.dataframe(
                comparison_df.round(4),
                width="stretch",
                hide_index=True
            )

            st.success(
                f"Overall model based on highest F1-Score: "
                f"**{best_model}**"
            )


        # ====================================================
        # COMPLETION
        # ====================================================

        st.success(
            f"{selected_model_name} prediction completed successfully!"
        )


else:

    st.info(
        "Please upload test_data.csv to begin prediction."
    )