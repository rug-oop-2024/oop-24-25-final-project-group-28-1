import streamlit as st
import pandas as pd
import requests
from os import listdir
from os.path import isfile, join
from pathlib import Path

from ST.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.write("**Developers info**")
st.write("This is ST/datasets/management/delete_dataset.py")
st.write("AutoMLSystem has been imported from core/system/py")
#  st.write("Dataset imported from  **autoop.core.ml.dataset**")
st.write("""\n\n Mission: From the list with datasets, delete one.""")
st.write("-"*80)

st.session_state.widget = ""

if "store_selection" not in st.session_state:
    st.session_state.store_selection = ""

if "selection" not in st.session_state:
    st.session_state.selection = None
    
def clear_box():
    st.session_state.store_selection = st.session_state.selection
    st.session_state.selection = None

#my_text = st.session_state.my_text
#st.write(my_text)   

if 'dataset_index' not in st.session_state:
    st.session_state.dataset_index = None
        
automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

selects = []
for ds in datasets:
    selects.append(ds.name)
 

st.write("### Select a dataset to delete")
prompt = "Select a dataset to delete:"    
delete_index = st.selectbox(prompt, range(len(selects)), 
                            format_func=lambda x: selects[x], 
                            index=None, 
                            key="selection")
                            #on_change=clear_box)
if not delete_index is None:
    if st.button("Delete"):
        st.write(datasets[delete_index])
        ds_name = datasets[delete_index].name
        ds_id = datasets[delete_index].id
        st.success(f"Deleted {ds_name} ({ds_id})")
        # st.session_state.selection = None
        
    
# After deleting sets, the identification with an index number is not valid anymore
# if you throw away datasets. So set its state to None
st.session_state.dataset_index = None
