import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def climate_show():
    st.header("Exploratory Data Analysis (EDA)")

    # Climate and Weather data
    st.subheader('Climate and Weather data trend of Nepal')
    # Load data
    df = pd.read_csv(
        "data\weatherandclimatedata\combined_data.csv")

    # Set style
    sns.set(style="whitegrid")

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Temperature trends
    sns.lineplot(data=df, x="year", y="avg_mean_temp", ax=axs[0, 0], label="Mean Temp")
    sns.lineplot(data=df, x="year", y="avg_min_temp", ax=axs[0, 0], label="Min Temp")
    sns.lineplot(data=df, x="year", y="avg_max_temp", ax=axs[0, 0], label="Max Temp")
    axs[0, 0].set_title("Temperature Trends Over Time")
    axs[0, 0].legend()

    # Humidity and rainfall
    sns.lineplot(data=df, x="year", y="relative_humidity", ax=axs[0, 1], label="Humidity")
    sns.lineplot(data=df, x="year", y="annual_rainfall", ax=axs[0, 1], label="Rainfall")
    axs[0, 1].set_title("Humidity and Rainfall Over Time")
    axs[0, 1].legend()

    # Agricultural indicators
    sns.lineplot(data=df, x="year", y="agri_land_area", ax=axs[1, 0], label="Agri Land Area")
    sns.lineplot(data=df, x="year", y="cropland_pct", ax=axs[1, 0], label="Cropland %")
    axs[1, 0].set_title("Agricultural Land Use Over Time")
    axs[1, 0].legend()

    # Fertilizer use and population density
    sns.lineplot(data=df, x="year", y="fertilizer_kg_per_ha", ax=axs[1, 1], label="Fertilizer Use")
    sns.lineplot(data=df, x="year", y="population_density", ax=axs[1, 1], label="Population Density")
    axs[1, 1].set_title("Fertilizer Use and Population Density Over Time")
    axs[1, 1].legend()

    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)

def env_show():
    st.subheader("Environmental data visualization of Nepal")

    # Load data
    env_df = pd.read_csv('data/environmentaldata/forest-area.csv')

    # Calculate forest land percentage
    env_df["Forest %"] = (env_df["Forest land"] / env_df["Total Land"]) * 100

    sns.set(style="whitegrid")

    # First figure: Top 10 districts by highest forest coverage
    top10_high = env_df.sort_values("Forest %", ascending=False).head(10)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top10_high, x="Forest %", y="District", palette="Greens_r", ax=ax1)
    ax1.set_title("Top 10 Districts by Forest Coverage (%)")
    st.pyplot(fig1)

    # Second figure: Top 10 districts by least forest coverage
    top10_low = env_df.sort_values("Forest %", ascending=True).head(10)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top10_low, x="Forest %", y="District", palette="Reds_r", ax=ax2)
    ax2.set_title("Top 10 Districts by Least Forest Coverage (%)")
    st.pyplot(fig2)

def socio_show():
    st.subheader("Socio-Economic data visualization of Nepal")
    sco_eco_df = pd.read_csv('data\socioeconomicdata\eco-socio-env-health-edu-dev-energy_npl.csv')
    # Basic summary statistics for 'Value'
    summary_stats = sco_eco_df['Value'].describe()
    # Count of each unique indicator
    indicator_counts = sco_eco_df['Indicator Name'].value_counts().head(20)

    indicator_df = indicator_counts.reset_index()
    indicator_df.columns = ['Indicator Name', 'Count']

    fig3 = plt.figure(figsize=(10, 8))
    sns.barplot(
        data=indicator_df,
        y='Indicator Name',
        x='Count',
        hue='Indicator Name',
        palette='viridis',
        dodge=False,
        legend=False
    )
    plt.title("Top 20 Most Recorded Indicators")
    plt.xlabel("Number of Records")
    plt.ylabel("Indicator Name")
    plt.tight_layout()
    plt.grid(True, axis='x')
    plt.show()

    st.pyplot(fig3)

    st.subheader('Time series plot')
    # Pick 4 most frequent indicators for time-series visualization
    top_indicators = sco_eco_df['Indicator Name'].value_counts().head(4).index.tolist()

    # Filter data for those indicators
    df_top_indicators = sco_eco_df[sco_eco_df['Indicator Name'].isin(top_indicators)]

    # Plotting
    fig4 = plt.figure(figsize=(14, 10))
    for i, indicator in enumerate(top_indicators, 1):
        plt.subplot(2, 2, i)
        subset = df_top_indicators[df_top_indicators['Indicator Name'] == indicator]
        sns.lineplot(data=subset, x='Year', y='Value')
        plt.title(indicator)
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    st.pyplot(fig4)