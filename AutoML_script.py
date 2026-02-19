"""
🚀AutoML Builder
Author: Gaurav
Purpose:
- Upload data
- Explore it visually
- Clean it consciously
- Detect anomalies responsibly
- Train the best ML model automatically

Designed to be readable, explainable, and interview-ready.
"""

import streamlit as st
# Add custom CSS to hide the GitHub icon
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your app code goes here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ML utilities
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    IsolationForest
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, r2_score

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="AutoML Builder", layout="wide")
st.title("🚀 No-Code AutoML Builder")
st.caption("EDA → Cleaning → Anomaly Detection → Modeling")

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

if uploaded_file:

    # Load data
    raw_data = pd.read_csv(uploaded_file)

    st.subheader("👀 Quick Data Preview")
    st.dataframe(raw_data.head())

    # =========================================================
    # 🧹 DATA CLEANING (USER-CONTROLLED)
    # =========================================================
    st.header("🧹 Data Cleaning")

    cleaning_strategy = st.selectbox(
        "How should missing values be handled?",
        [
            "Do nothing",
            "Drop rows with missing values",
            "Impute Mean",
            "Impute Median",
            "Impute Min",
            "Impute Max",
            "Impute Most Frequent"
        ]
    )

    remove_zero_rows = st.checkbox(
        "Remove rows containing zero values (numeric columns only)"
    )

    cleaned_data = raw_data.copy()

    # ---- Handle missing values ----
    if cleaning_strategy == "Drop rows with missing values":
        cleaned_data.dropna(inplace=True)

    elif cleaning_strategy != "Do nothing":
        for column in cleaned_data.columns:
            if cleaned_data[column].dtype != "object":
                if cleaning_strategy == "Impute Mean":
                    cleaned_data[column].fillna(cleaned_data[column].mean(), inplace=True)
                elif cleaning_strategy == "Impute Median":
                    cleaned_data[column].fillna(cleaned_data[column].median(), inplace=True)
                elif cleaning_strategy == "Impute Min":
                    cleaned_data[column].fillna(cleaned_data[column].min(), inplace=True)
                elif cleaning_strategy == "Impute Max":
                    cleaned_data[column].fillna(cleaned_data[column].max(), inplace=True)
            else:
                cleaned_data[column].fillna(cleaned_data[column].mode()[0], inplace=True)

    # ---- Remove zero rows ----
    if remove_zero_rows:
        numeric_columns = cleaned_data.select_dtypes(include="number").columns
        cleaned_data = cleaned_data[(cleaned_data[numeric_columns] != 0).all(axis=1)]

    st.success(f"✅ Cleaned dataset shape: {cleaned_data.shape}")

    # =========================================================
    # 🔍 EXPLORATORY DATA ANALYSIS
    # =========================================================
    st.header("🔍 Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", cleaned_data.shape[0])
    col2.metric("Columns", cleaned_data.shape[1])
    col3.metric(
        "Missing %",
        round(cleaned_data.isna().mean().mean() * 100, 2)
    )

    numeric_cols = cleaned_data.select_dtypes(include="number").columns.tolist()
    categorical_cols = cleaned_data.select_dtypes(exclude="number").columns.tolist()

    if st.button("📦 Show Boxplots (Outlier Preview)"):
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=cleaned_data[col], ax=ax)
            ax.set_title(f"Boxplot: {col}")
            st.pyplot(fig)

    # =========================================================
    # 🚨 ANOMALY DETECTION (PRE-MODEL)
    # =========================================================
    st.header("🚨 Anomaly Detection")

    anomaly_method = st.selectbox(
        "Choose anomaly detection method",
        ["None", "Isolation Forest", "Local Outlier Factor", "Z-Score"]
    )

    contamination_rate = st.slider(
        "Expected % of anomalies",
        min_value=1,
        max_value=20,
        value=5
    ) / 100

    drop_anomalies = st.checkbox("Remove detected anomalies before modeling")

    anomaly_ready_data = cleaned_data.copy()

    if anomaly_method != "None" and len(numeric_cols) > 0:

        numeric_data = cleaned_data[numeric_cols]

        if anomaly_method == "Isolation Forest":
            detector = IsolationForest(
                contamination=contamination_rate,
                random_state=42
            )
            anomaly_ready_data["anomaly_flag"] = detector.fit_predict(numeric_data)

        elif anomaly_method == "Local Outlier Factor":
            detector = LocalOutlierFactor(contamination=contamination_rate)
            anomaly_ready_data["anomaly_flag"] = detector.fit_predict(numeric_data)

        elif anomaly_method == "Z-Score":
            z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
            anomaly_ready_data["anomaly_flag"] = np.where(
                (z_scores > 3).any(axis=1), -1, 1
            )

        anomaly_count = (anomaly_ready_data["anomaly_flag"] == -1).sum()
        st.warning(f"⚠️ Detected anomalies: {anomaly_count}")

        if st.button("📊 Visualise Anomalies"):
            if len(numeric_cols) >= 2:
                fig, ax = plt.subplots()
                sns.scatterplot(
                    x=anomaly_ready_data[numeric_cols[0]],
                    y=anomaly_ready_data[numeric_cols[1]],
                    hue=anomaly_ready_data["anomaly_flag"],
                    palette={1: "blue", -1: "red"},
                    ax=ax
                )
                ax.set_title("Anomaly Distribution")
                st.pyplot(fig)

        if drop_anomalies:
            cleaned_data = (
                anomaly_ready_data[anomaly_ready_data["anomaly_flag"] == 1]
                .drop(columns="anomaly_flag")
            )
            st.success(f"🧹 Data after anomaly removal: {cleaned_data.shape}")

    # =========================================================
    # 🤖 AutoML MODEL BUILDER
    # =========================================================
    st.header("🤖 AutoML Model Builder")

    target_column = st.selectbox("Select target variable", cleaned_data.columns)

    X = cleaned_data.drop(columns=[target_column])
    y = cleaned_data[target_column]

    problem_type = st.radio("Problem type", ["Regression", "Classification"])
    scaling_choice = st.selectbox("Scaling method", ["None", "Standard", "MinMax"])

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    # ---- Numeric pipeline ----
    numeric_pipeline = [("imputer", SimpleImputer(strategy="median"))]

    if scaling_choice == "Standard":
        numeric_pipeline.append(("scaler", StandardScaler()))
    elif scaling_choice == "MinMax":
        numeric_pipeline.append(("scaler", MinMaxScaler()))

    # ---- Full preprocessing ----
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_pipeline), numeric_features),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_features
            )
        ]
    )

    # ---- Model zoo ----
    models = (
        {
            "Linear Regression": (LinearRegression(), {}),
            "Random Forest Regressor": (
                RandomForestRegressor(),
                {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]}
            )
        }
        if problem_type == "Regression"
        else
        {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000),
                {"model__C": [0.1, 1, 10]}
            ),
            "Random Forest Classifier": (
                RandomForestClassifier(),
                {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]}
            )
        }
    )

    if st.button("🚀 Run AutoML"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_scores = {}
        best_model = None
        best_score = -np.inf

        for model_name, (model, params) in models.items():
            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            search = GridSearchCV(
                pipeline,
                params,
                cv=3,
                scoring="r2" if problem_type == "Regression" else "accuracy"
            )

            search.fit(X_train, y_train)
            predictions = search.predict(X_test)

            score = (
                r2_score(y_test, predictions)
                if problem_type == "Regression"
                else accuracy_score(y_test, predictions)
            )

            model_scores[model_name] = score

            if score > best_score:
                best_score = score
                best_model = search.best_estimator_

        st.subheader("🏆 Model Performance")
        st.bar_chart(model_scores)
        st.success(f"Best model score: {round(best_score, 3)}")

        joblib.dump(best_model, "best_model.pkl")
        with open("best_model.pkl", "rb") as file:
            st.download_button(
                "⬇️ Download Best Model",
                file,
                file_name="best_model.pkl"
            )
