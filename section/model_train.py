
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def model_train_and_evaluate():
    st.subheader("ML Model Training and Evaluation")

    # Load data
    df = pd.read_csv("data/weatherandclimatedata/combined_data.csv")

    # Drop year column and define features/target for regression
    X = df.drop(columns=["year", "annual_rainfall"])
    y = df["annual_rainfall"]

    st.markdown("### Regression Model - Predict Annual Rainfall")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

    # Plot actual vs predicted
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted Rainfall")
    st.pyplot(fig1)

    # Classification prep
    st.markdown("### Classification Model - Rainfall Level Category")
    df["rainfall_level"] = pd.qcut(df["annual_rainfall"], q=3, labels=["Low", "Medium", "High"])
    X_cls = df.drop(columns=["year", "annual_rainfall", "rainfall_level"])
    y_cls = df["rainfall_level"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(Xc_train, yc_train)
    yc_pred = clf.predict(Xc_test)

    st.text("Classification Report:")
    st.text(classification_report(yc_test, yc_pred))

    # Confusion Matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    fig2, ax2 = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(clf, Xc_test, yc_test, ax=ax2)
    st.pyplot(fig2)

    st.markdown("✅ Both regression and classification models are trained using the structured dataset.")
