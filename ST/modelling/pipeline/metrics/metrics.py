import streamlit as st

# Display header
st.write("This is: modelling/pipeline/metrics/metrics.py")
st.write("Prompt the user to select a set of compatible metrics.")

# Define available metrics for each task type
classification_metrics = ["Accuracy", "Precision", "Recall"]
regression_metrics = ["Mean Squared Error", "Mean Absolute Error", "R-Squared"]

# Prompt user to select task type
task_type = st.selectbox("Select the task type:",
                         ["Classification", "Regression"])

# Display compatible metrics based on selected task type
if task_type == "Classification":
    selected_metrics = st.multiselect(
        "Select metrics for classification:", classification_metrics)
elif task_type == "Regression":
    selected_metrics = st.multiselect(
        "Select metrics for regression:", regression_metrics)

# Show selected metrics
if selected_metrics:
    st.write("You selected the following metrics:")
    for metric in selected_metrics:
        st.write(f"- {metric}")
else:
    st.write("No metrics selected.")
