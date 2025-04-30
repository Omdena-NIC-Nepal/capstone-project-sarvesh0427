import streamlit as st

def show():
    st.header("Climate Change Impact Assessment and Prediction System for Nepal")
    st.subheader("About the Project")
    st.write("""
        **Climate Change Impact Assessment and Prediction System for Nepal** is a capstone project under the Omdena-NIC 
        Data Science and Machine Learning course. The objective is to analyze the effects of climate change across Nepal 
        using real-world environmental and climatic datasets.

        This dashboard presents key insights from datasets including:
        - Climate indicators (temperature, rainfall, humidity, etc.)
        - Environmental data (forest area by district)

        The system aims to support data-driven decisions for sustainable environmental planning and climate resilience in Nepal.
        """)
    st.subheader("Contributor")
    st.markdown("""
        - Sarvesh Chhetri  
        (Omdena’s & NIC Capacity Building Batch II)
        """)

    st.subheader("!!! Disclaimer")
    st.write("This project is for educational and research purposes only.")
