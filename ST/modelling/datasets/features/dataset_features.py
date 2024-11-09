import ST.helpermods.helper as helper
from ST.core.system import AutoMLSystem
from autoop.core.ml import feature
import streamlit as st
import numpy as np


st.write("**Developers info**")
st.write("modelling/datasets/features/dataset_features.py")
st.write("""\n\n Detect the features and generate a selection menu for
selecting the input
features (many) and one target feature. Based on the feature selections,
prompt the user
with the detected task type (i.e., classification or regression).""")
st.write("-"*4)

st.write("## Detect features")
dataset, automl = helper.get_selected_dataset()
if not dataset:
     st.stop()

s = f"Selected dataset: {dataset.name}"
st.write(s)

df = helper.dataset_to_pd(dataset)
num_cols = len(df.columns)


# Loop over all avaialble columns in the dataset.
# Then try to calculate the average. If that fails,
# the column is marked as 'categorical' else it
# is considered to be 'numerical'
# Create a streamlit form that allows a user to enter 
# the names of the columns
colnames = []
coltypes = []
with st.form("Features"):
     for i, df_col in enumerate(df.columns):
          #colnames.append(st.text_input(f"Name for column {i+1}"))
          try:
               av = df[df_col].mean()
               txt = "numerical"
               st.write(txt)               
          except:
               txt = "categorical"
               st.write(txt)               
          
          s = f"{i+1}: Name of {txt} column"
          colnames.append(st.text_input(s))
          coltypes.append(txt)
     submitted = st.form_submit_button("Store")

if not submitted:
     st.stop()     


# Loop again over all columns. Create Feature objects, calculate
# properties and store these in Feature attributes.
# Finally, create a dict with those attributes and store this
# in a dict with all features. Then re-register the modified 
# dataset so that the features can be retrieved later
feature_dict = {}
for i, df_col in enumerate(df.columns):
     feat = feature.Feature(name=colnames[i], type=coltypes[i])
     feat.calculate_statistics(df[df_col])
     feature_dict[feat.name] = feat.__dict__


dataset.add_features(feature_dict)
automl.registry.register(dataset)
st.write("Added Features:", dataset.get_features())

#st.write(df)
#st.write(feature1)
