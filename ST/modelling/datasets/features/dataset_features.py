import streamlit as st
from pandas import api
from autoop.core.ml import feature
import ST.helpermods.helper as helper

st.write("## Detect features")
dataset, automl = helper.get_selected_dataset()
if not dataset:
    st.stop()

s = f"Selected dataset: {dataset.name}"
st.info(s)

df = helper.dataset_to_pd(dataset)
num_cols = len(df.columns)

try:
    feats = list(dataset.features.keys())
except AttributeError:
    feats = None

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
        txt = "numerical" if api.types.is_numeric_dtype(
             df[df_col]
          ) else "categorical"

        s = f"Feature {i + 1}: Name of **{txt}** column"
        default_colname = colnames[i] if names_exist else f"column{i + 1}"
        colname = st.text_input(s, value=default_colname)

        if names_exist:
            colnames[i] = colname
        else:
            colnames.append(colname)

        coltypes.append(txt)

    submitted = st.form_submit_button("Store")

if not submitted:
    st.stop()

feature_dict = {}
for i, df_col in enumerate(df.columns):
    feat = feature.Feature(name=colnames[i], type=coltypes[i])
    feat.calculate_statistics(df[df_col])
    feature_dict[feat.name] = feat.__dict__

dataset.add_features(feature_dict)
automl.registry.register(dataset)
st.write("## Added Features:", dataset.get_features())
