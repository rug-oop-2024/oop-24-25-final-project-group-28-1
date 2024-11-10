import ST.helpermods.helper as helper
from ST.core.system import AutoMLSystem
from autoop.core.ml import feature
import streamlit as st
import numpy as np
from pandas import api


#st.write("**Developers info**")
#st.write("modelling/datasets/features/dataset_features.py")
#st.write("""\n\n Detect the features and generate a selection menu for
#selecting the input
#features (many) and one target feature. Based on the feature selections,
#prompt the user
#with the detected task type (i.e., classification or regression).""")
#st.write("-"*4)

st.write("## Detect features")
dataset, automl = helper.get_selected_dataset()
if not dataset:
     st.stop()

s = f"Selected dataset: {dataset.name}"
st.info(s)

df = helper.dataset_to_pd(dataset)
num_cols = len(df.columns)

# Loop over all avaialble columns in the dataset.
# Then try to calculate the average. If that fails,
# the column is marked as 'categorical' else it
# is considered to be 'numerical'
# Create a streamlit form that allows a user to enter 
# the names of the columns

feats = list(dataset.features.keys())
if feats:
     colnames = feats
     names_exist = True
else:
     colnames = []
     names_exist = False

st.write("### Enter column names")
coltypes = []
with st.form("Features"):
     for i, df_col in enumerate(df.columns):
          #colnames.append(st.text_input(f"Name for column {i+1}"))
          if api.types.is_numeric_dtype(df[df_col]):
               txt = "numerical"
          else:
               txt = "categorical"          
          s = f"Feature {i+1}: Name of **{txt}** column"
          if names_exist:
              default_colname = colnames[i]
          else:
               default_colname = f"column{i+1}"
          colname = st.text_input(s, value=default_colname)
          if names_exist:
               colnames[i] = colname 
          else:
               colnames.append(colname)
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
st.write("## Added Features:", dataset.get_features())

