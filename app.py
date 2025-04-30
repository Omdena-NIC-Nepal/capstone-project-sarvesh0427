import streamlit as st

# Page config - wide layout, no sidebar
st.set_page_config(page_title="Nepal Climate", layout="wide")

# Sidebar tabs
selected = st.sidebar.radio(
    "Explore",
    ["About the project", "Exploratory Data Analysis", "Model train and Evaluation"],
    index=1
)

if selected == "About the project":
    from section import about
    about.show()

# Exploratory Data Analysis section
elif selected == "Exploratory Data Analysis":
    from section import eda
    eda.climate_show()
    eda.env_show()
    eda.socio_show()


elif selected == "Model train and Evaluation":
    from section import model_train
    model_train.model_train_and_evaluate()

# Horizontal line
st.markdown("---")
# Centered footer using Markdown hack
st.markdown(
    "<p style='text-align: center;'>© 2025 Omdena-NIC Capstone Project | Climate Change Impact Assessment - Nepal</p>",
    unsafe_allow_html=True
)
