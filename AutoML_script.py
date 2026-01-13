import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

from fpdf import FPDF

# ---------------- CONFIG ----------------
st.set_page_config("AutoML Builder", layout="wide")
st.title("🚀 No-Code AutoML Builder with EDA & PDF Export")

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # ---------------- EDA SECTION ----------------
    st.header("🔍 Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing %", round(df.isna().mean().mean()*100, 2))

    st.subheader("Column Types")
    st.write(df.dtypes)

    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) > 0:
        feature = st.selectbox("Distribution Plot", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[num_cols].corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ---------------- PDF EXPORT ----------------
    if st.button("📥 Download EDA PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)

        pdf.cell(200, 10, txt="EDA Report", ln=True)

        pdf.cell(200, 8, txt=f"Rows: {df.shape[0]}", ln=True)
        pdf.cell(200, 8, txt=f"Columns: {df.shape[1]}", ln=True)
        pdf.cell(200, 8, txt=f"Missing %: {round(df.isna().mean().mean()*100,2)}", ln=True)

        pdf.output("eda_report.pdf")

        with open("eda_report.pdf", "rb") as f:
            st.download_button("Download PDF", f, "eda_report.pdf")

    # ---------------- MODEL BUILDER ----------------
    st.header("🤖 AutoML Model Builder")

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    task = st.radio("Task Type", ["Regression", "Classification"])

    scale = st.selectbox("Scaling Method", ["None", "Standard", "MinMax"])

    num_features = X.select_dtypes(include="number").columns.tolist()
    cat_features = X.select_dtypes(exclude="number").columns.tolist()

    # ---------------- PREPROCESSOR ----------------
    num_pipeline = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale == "Standard":
        num_pipeline.append(("scaler", StandardScaler()))
    elif scale == "MinMax":
        num_pipeline.append(("scaler", MinMaxScaler()))

    preprocessor = ColumnTransformer([
        ("num", Pipeline(num_pipeline), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features)
    ])

    # ---------------- MODELS + HYPERPARAMS ----------------
    if task == "Regression":
        models = {
            "Linear Regression": {
                "model": LinearRegression(),
                "params": {}
            },
            "Random Forest": {
                "model": RandomForestRegressor(),
                "params": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [None, 10, 20]
                }
            }
        }
    else:
        models = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=1000),
                "params": {
                    "model__C": [0.1, 1, 10]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(),
                "params": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [None, 10]
                }
            }
        }

    # ---------------- TRAIN ----------------
    if st.button("🚀 Run AutoML"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        results = {}
        best_model = None
        best_score = -999

        for name, config in models.items():
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", config["model"])
            ])

            gs = GridSearchCV(
                pipe,
                config["params"],
                cv=3,
                scoring="r2" if task == "Regression" else "accuracy"
            )

            gs.fit(X_train, y_train)
            preds = gs.predict(X_test)

            score = r2_score(y_test, preds) if task == "Regression" else accuracy_score(y_test, preds)
            results[name] = score

            if score > best_score:
                best_score = score
                best_model = gs.best_estimator_

        st.subheader("🏆 Model Comparison")
        st.bar_chart(results)

        st.success(f"Best Model Score: {round(best_score,3)}")

        # ---------------- DOWNLOAD MODEL ----------------
        joblib.dump(best_model, "best_model.pkl")
        with open("best_model.pkl", "rb") as f:
            st.download_button("⬇ Download Best Model", f, "best_model.pkl")
