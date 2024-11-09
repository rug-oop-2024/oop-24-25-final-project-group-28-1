import streamlit as st
from ST.core.system import AutoMLSystem
import pandas as pd
import io


def get_selected_dataset():
     if 'dataset_index' not in st.session_state:
         st.write("No dataset is selected")
         st.session_state.dataset_index = None
         st.warning('No dataset selected', icon="⚠️")
         return None, None

     elif st.session_state.dataset_index:
         index = int(st.session_state.dataset_index)
         s = f"Index selected dataset: {index}"
         st.write(s)
         automl = AutoMLSystem.get_instance()
         dataset = automl.registry.list(type="dataset")[index]
         return dataset, automl
     
     return None, None
 

def dataset_to_pd(dataset):
    data = dataset.data.decode()     # Should be using dataset.read() but this returns the bytes object only.
                                     # TODO Find what must be done to be able to use the read method.
    retrieved_df = pd.read_csv(io.StringIO(data))
    return retrieved_df
