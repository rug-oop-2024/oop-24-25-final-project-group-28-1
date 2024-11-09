import streamlit as st
import ST.helpermods.helper as helper
from ST.core.system import AutoMLSystem


st.write("**Developers info**")
st.write("modelling/datasets/features/dataset_features.py")
st.write("""\n\n Detect the features and generate a selection menu for
selecting the input
features (many) and one target feature. Based on the feature selections,
prompt the user
with the detected task type (i.e., classification or regression).""")
st.write("-"*80)

st.write("## Detect features")
dataset = helper.get_selected_dataset()
if not dataset:
     st.stop()

s = f"Selected dataset: {dataset.name}"
st.write(s)