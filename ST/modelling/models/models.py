import streamlit as st


# Developers info
#st.write("This is: modelling/models/models.py")
#st.write("\n\nPrompt the user to select a model based on the task type.")

st.write("Model selection")

selects = ["Logistic Regresssion", "Random forest classifier", "SVM classifier",
           "Multiple linear regression", "Random forest regressor", "Ridge regression"]

prompt = "Model:"
helpmes = "Select a model, appropriate for your data"
selected_model = st.selectbox(
    prompt,
    selects,
    index=None,
    placeholder="Select a model from this drop down menu",
    help=helpmes,
)

if selected_model:
    s = f"Selected model: {selected_model}"
    st.success(s)
if 'selected_model' not in st.session_state or not selected_model:
    st.session_state.selected_model = None
else:
    st.session_state.selected_model= selected_model
