import streamlit as st

from ST.core.system import AutoMLSystem


def get_selected_dataset():
    """Getter for dataset."""
    if "dataset_index" not in st.session_state:
        st.write("No dataset is selected")
        st.session_state.dataset_index = None
        st.warning("No dataset selected", icon="⚠️")
        return None

    elif st.session_state.dataset_index:
        index = int(st.session_state.dataset_index)
        text = f"Index selected dataset: {index}"
        st.write(text)
        automl = AutoMLSystem.get_instance()
        dataset = automl.registry.list(type="dataset")[index]
        return dataset

    return None
