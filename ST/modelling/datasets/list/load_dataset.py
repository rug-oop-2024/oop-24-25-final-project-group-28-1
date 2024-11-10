import io
import pandas as pd
import streamlit as st
from ST.core.system import AutoMLSystem

st.write("## Loading dataset")
dataset = None
if 'dataset_index' not in st.session_state:
    st.session_state.dataset_index = None

# An index for a dataset is not yet known, so we cannot use
# the helper function get_selected_dataset()
automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

selects = [ds.name for ds in datasets]

prompt = "Select a data set:"
index = st.selectbox(
    prompt,
    range(len(selects)),
    format_func=lambda x: selects[x],
    index=None
)
if index is not None:
    dataset = datasets[index]
    data = dataset.data.decode()  # Decoding bytes to string
    # TODO: Find what must be done to be able to use the read method.

    retrieved_df = pd.read_csv(io.StringIO(data))
    st.success(f"The selected dataset is {dataset.name}")
    st.write("### Properties:")
    st.write(retrieved_df)
    st.write("Features:", dataset.get_features())

st.session_state.dataset_index = str(index)
