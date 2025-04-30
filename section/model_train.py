import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def model_train_and_evaluate():
    st.subheader("ML Model Training and Evaluation")

    # Load data
    df = pd.read_csv("data/weatherandclimatedata/combined_data.csv")

    # Regression setup
    X = df.drop(columns=["year", "annual_rainfall"])
    y = df["annual_rainfall"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Classification setup
    df["rainfall_level"] = pd.qcut(df["annual_rainfall"], q=3, labels=["Low", "Medium", "High"])
    X_cls = df.drop(columns=["year", "annual_rainfall", "rainfall_level"])
    y_cls = df["rainfall_level"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(Xc_train, yc_train)
    yc_pred = clf.predict(Xc_test)

    st.markdown("Predict Future Rainfall (User Input)")

    # User inputs
    avg_temp = st.slider("Average Mean Temp (°C)", 0.0, 40.0, 15.0)
    min_temp = st.slider("Average Min Temp (°C)", -5.0, 30.0, 5.0)
    max_temp = st.slider("Average Max Temp (°C)", 5.0, 50.0, 25.0)
    humidity = st.slider("Relative Humidity (%)", 0.0, 100.0, 70.0)
    precip_max = st.slider("Max Precipitation (mm)", 0.0, 500.0, 100.0)
    agri_area = st.slider("Agri Land Area (sq km)", 0.0, 100000.0, 50000.0)
    cropland_pct = st.slider("Cropland %", 0.0, 1.0, 0.5)
    fert_kg_ha = st.slider("Fertilizer (kg/ha)", 0.0, 300.0, 50.0)
    pop_density = st.slider("Population Density (per sq km)", 0.0, 1000.0, 200.0)

    # Prediction trigger
    if st.button("Predict"):
        input_data = pd.DataFrame([[
            avg_temp, min_temp, max_temp, humidity, precip_max,
            agri_area, cropland_pct, fert_kg_ha, pop_density
        ]], columns=X.columns)

        future_rainfall = model.predict(input_data)[0]
        future_class = clf.predict(input_data)[0]

        st.success(f"🌧️ **Predicted Annual Rainfall:** {future_rainfall:.2f} mm")
        st.info(f"📊 **Rainfall Category:** {future_class}")

    st.markdown('---')
    st.subheader("Visualization of training datasets prediction")
    st.markdown("Regression Model - Predict Annual Rainfall")
    st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted Rainfall")
    st.pyplot(fig1)


    st.markdown("Classification Model - Rainfall Level")
    st.text(classification_report(yc_test, yc_pred))
    fig2, ax2 = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(clf, Xc_test, yc_test, ax=ax2)
    st.pyplot(fig2)

