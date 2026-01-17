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
st.title("🚀 No-Code AutoML Builder with Advanced EDA")

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ---------------- DATA CLEANING ----------------
    st.header("🧹 Data Cleaning")

    cleaning_method = st.selectbox(
        "Handle Missing Values",
        [
            "None",
            "Drop rows with nulls",
            "Impute Mean",
            "Impute Median",
            "Impute Min",
            "Impute Max",
            "Impute Most Frequent"
        ]
    )

    remove_zero = st.checkbox("Remove rows with zero values (numeric only)")

    df_clean = df.copy()

    if cleaning_method == "Drop rows with nulls":
        df_clean = df_clean.dropna()

    elif cleaning_method.startswith("Impute"):
        for col in df_clean.columns:
            if df_clean[col].dtype != "object":
                if cleaning_method == "Impute Min":
                    df_clean[col].fillna(df_clean[col].min(), inplace=True)
                elif cleaning_method == "Impute Max":
                    df_clean[col].fillna(df_clean[col].max(), inplace=True)
                elif cleaning_method == "Impute Mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif cleaning_method == "Impute Median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    if remove_zero:
        num_cols = df_clean.select_dtypes(include="number").columns
        df_clean = df_clean[(df_clean[num_cols] != 0).all(axis=1)]

    st.success(f"Cleaned Dataset Shape: {df_clean.shape}")

    # ---------------- BASIC EDA ----------------
    st.header("🔍 Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df_clean.shape[0])
    col2.metric("Columns", df_clean.shape[1])
    col3.metric("Missing %", round(df_clean.isna().mean().mean()*100, 2))

    num_cols = df_clean.select_dtypes(include="number").columns.tolist()
    cat_cols = df_clean.select_dtypes(exclude="number").columns.tolist()

    # ---------------- ADVANCED EDA (BUTTONS) ----------------
    if st.button("📌 Missing Values Plot"):
        fig, ax = plt.subplots()
        df_clean.isna().sum().plot(kind="bar", ax=ax)
        ax.set_title("Missing Values")
        st.pyplot(fig)

    if st.button("📦 Boxplots (Outliers)"):
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df_clean[col], ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

    if st.button("🔗 Pairplot (Sampled)"):
        sample_df = df_clean[num_cols].sample(min(300, len(df_clean)))
        fig = sns.pairplot(sample_df)
        st.pyplot(fig)

    if st.button("📊 Categorical Value Counts"):
        for col in cat_cols:
            fig, ax = plt.subplots()
            df_clean[col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

    # ---------------- TARGET vs FEATURE ----------------
    st.header("🎯 Target vs Feature")

    target_eda = st.selectbox("Select Target", df_clean.columns)
    feature_eda = st.selectbox("Select Feature", df_clean.columns)

    if st.button("📈 Show Relationship"):
        fig, ax = plt.subplots()
        if df_clean[target_eda].dtype != "object":
            sns.scatterplot(x=df_clean[feature_eda], y=df_clean[target_eda], ax=ax)
        else:
            sns.boxplot(x=df_clean[feature_eda], y=df_clean[target_eda], ax=ax)
        st.pyplot(fig)

    # ---------------- CUSTOM VISUAL BUILDER ----------------
    st.header("🎨 Custom Visualization Builder")

    x_var = st.selectbox("X Variable", df_clean.columns)
    y_var = st.selectbox("Y Variable (Optional)", ["None"] + df_clean.columns.tolist())
    chart_type = st.selectbox(
        "Chart Type",
        ["Histogram", "Boxplot", "Scatter", "Bar", "Line"]
    )

    if st.button("🎬 Generate Chart"):
        fig, ax = plt.subplots()
        if chart_type == "Histogram":
            sns.histplot(df_clean[x_var], kde=True, ax=ax)
        elif chart_type == "Boxplot":
            sns.boxplot(x=df_clean[x_var], ax=ax)
        elif chart_type == "Scatter" and y_var != "None":
            sns.scatterplot(x=df_clean[x_var], y=df_clean[y_var], ax=ax)
        elif chart_type == "Bar" and y_var != "None":
            sns.barplot(x=df_clean[x_var], y=df_clean[y_var], ax=ax)
        elif chart_type == "Line" and y_var != "None":
            ax.plot(df_clean[x_var], df_clean[y_var])
        st.pyplot(fig)

    # ---------------- PDF EXPORT ----------------
    if st.button("📥 Download EDA PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, "EDA Report", ln=True)
        pdf.cell(200, 8, f"Rows: {df_clean.shape[0]}", ln=True)
        pdf.cell(200, 8, f"Columns: {df_clean.shape[1]}", ln=True)
        pdf.output("eda_report.pdf")

        with open("eda_report.pdf", "rb") as f:
            st.download_button("Download PDF", f, "eda_report.pdf")

    # ---------------- AutoML ----------------
    st.header("🤖 AutoML Model Builder")

    target = st.selectbox("Select Target Column", df_clean.columns)
    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    task = st.radio("Task Type", ["Regression", "Classification"])
    scale = st.selectbox("Scaling Method", ["None", "Standard", "MinMax"])

    num_features = X.select_dtypes(include="number").columns.tolist()
    cat_features = X.select_dtypes(exclude="number").columns.tolist()

    num_pipeline = [("imputer", SimpleImputer(strategy="median"))]
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

    models = (
        {
            "Linear Regression": (LinearRegression(), {}),
            "Random Forest": (RandomForestRegressor(), {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10]
            })
        }
        if task == "Regression"
        else
        {
            "Logistic Regression": (LogisticRegression(max_iter=1000), {
                "model__C": [0.1, 1, 10]
            }),
            "Random Forest": (RandomForestClassifier(), {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10]
            })
        }
    )

    if st.button("🚀 Run AutoML"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        results, best_score, best_model = {}, -999, None

        for name, (model, params) in models.items():
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            gs = GridSearchCV(
                pipe,
                params,
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
        st.success(f"Best Model Score: {round(best_score, 3)}")

        joblib.dump(best_model, "best_model.pkl")
        with open("best_model.pkl", "rb") as f:
            st.download_button("⬇ Download Best Model", f, "best_model.pkl")
