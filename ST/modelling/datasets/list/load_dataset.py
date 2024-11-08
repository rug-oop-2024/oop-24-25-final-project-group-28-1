from ST.core.system import AutoMLSystem
import streamlit as st
import pandas as pd
import io

st.write("**Developers info**")
st.write("This is modelling/datasets/list/load_dataset.py")
st.write("\n\n Load existing datasets using the artifact registry. You can use a select box to achieve this.")
st.write("---")

if 'dataset_index' not in st.session_state:
    st.session_state.dataset_index = None
        
automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

selects = []
for ds in datasets:
    selects.append(ds.name)
 
 
st.write("### Select a data set")
prompt = "Select a data set:"    
index = st.selectbox(prompt, range(len(selects)), format_func=lambda x: selects[x], index=None)
if not index is None:
    st.write("option:", selects[index])
    
    st.write("index:", index)
    dataset = datasets[index]
    # st.write(dataset.data)
    data = dataset.data.decode()     # Should be using dataset.read() but this returns the bytes object only.
                                     # TODO Find what must be done to be able to use the read method.
    retrieved_df = pd.read_csv(io.StringIO(data))
    #retrieved_df = dataset.read()
    st.write(retrieved_df)
    
st.session_state.dataset_index = str(index)
st.write(st.session_state)
