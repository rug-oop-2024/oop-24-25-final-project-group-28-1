import streamlit as st
from ST.core.system import AutoMLSystem


st.write("**Developers info**")
st.write("modelling/datasets/features/dataset_features.py")
st.write("""\n\n Detect the features and generate a selection menu for selecting the input 
features (many) and one target feature. Based on the feature selections, prompt the user 
with the detected task type (i.e., classification or regression).""")
st.write("-"*80)


if 'dataset_index' not in st.session_state:
    st.write("No dataset is selected")    
    st.session_state.dataset_index = None
if st.session_state.dataset_index:
    index = int(st.session_state.dataset_index)
    s = f"Index selected dataset: {index}"
    st.write(s)
    automl = AutoMLSystem.get_instance()
    dataset = automl.registry.list(type="dataset")[index]
    s = f"Selected dataset: {dataset.name}"
    st.write(s)
