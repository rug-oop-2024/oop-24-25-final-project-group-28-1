import streamlit as st

# Define available metrics for each task type
classification_metrics = ["Accuracy", "Precision", "Recall"]
regression_metrics = ["Mean Squared Error", "Mean Absolute Error", "R-Squared"]

# Prompt user to select task type
task_type = st.selectbox("Select the task type:", ["Classification", "Regression"])

# Initialize session state if not already present
if 'selected_metrics' not in st.session_state:
    st.session_state.selected_metrics = []

# Display compatible metrics based on selected task type and save them
if task_type == "Classification":
    selected_metrics = st.multiselect("Select metrics for classification:", classification_metrics, key="selected_metrics")
elif task_type == "Regression":
    selected_metrics = st.multiselect("Select metrics for regression:", regression_metrics, key="selected_metrics")

# Display selected metrics
st.success("You selected the following metrics:")
for metric in st.session_state.selected_metrics:
    st.success(f"- {metric}")
