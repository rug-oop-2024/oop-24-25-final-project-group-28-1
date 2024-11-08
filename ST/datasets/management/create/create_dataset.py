import streamlit as st
import pandas as pd
import requests
from os import listdir
from os.path import isfile, join
from pathlib import Path

from ST.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.write("**Developers info**")
st.write("This is ST/datasets/management/create_dataset.py")
st.write("AutoMLSystem has been imported from core/system/py")
st.write("Dataset imported from  **autoop.core.ml.dataset**")
st.write(
    """\n\n Mission: Upload a CSV dataset (e.g., Iris) and convert that into a dataset using the from_dataframe 
factory method. Since a dataset is already an artifact, you can use the **AutoMLSystem.get_instance** 
singelton class to to access either storage, database, or the artifact registry to save it."""
)
st.write("-" * 80)

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")
# st.write(f"Debug: datasets instance: {datasets[0]}")

if "selectbox" not in st.session_state:
    st.session_state.selectbox = None

def csv2pd(path_or_url, headertype=None):
    if 1:  # try:
        df = pd.read_csv(path_or_url, sep=",", header=headertype)
        return df
    else:  # except:
        st.write("Not a valid URL:", path_or_url)
        return None


st.write("## Open a dataset")
st.write("### 1. From URL:")
iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
st.write(f"Enter url (e.g. {iris_url}): ")
url = st.text_input("Enter URL or path to file:")

st.write("### 2. From storage:")
# Get files in ./assets/objects
storagepath = Path("./assets/objects/")
csvfiles = ["None"] + list(storagepath.glob("*.csv"))
helpmes = """Open this drop down menu and\n select one of the available data sets."""
stored_dataset = st.selectbox(
    "or select an existing file:",
    csvfiles,
    index=None,
    placeholder="Select a file from this drop down menu",
    help=helpmes,
    key="selectbox",
)
if stored_dataset == "None":
    stored_dataset = None

st.write("### 3. File uploader (from local file system):")
uploaded_file = st.file_uploader(
    "or upload a file", type="csv"
)  # Sets the file manager's filter to ".csv"
if uploaded_file:
    url = ""
    stored_dataset = ""

# VOG: as a test(!) I stored some CSV files in ./assets/objects
# These files can be presented in a drop down menu and one can be
# selected by the user. Note that this part must be replaced by
# an OOP method with a database for example.

path = None
if url:
    st.write("Dataset from url:", url)
    path = url
elif stored_dataset:
    st.write("From local storage:", stored_dataset)
    path = stored_dataset
elif uploaded_file:
    st.write("Uploaded file:", uploaded_file.name)  # upload_file object has attributes
    path = uploaded_file
else:
    st.write("Nothing selected yet:")

has_header = st.checkbox("Has header?")
headertype = "infer" if has_header else None

df = None
if path:
    df = csv2pd(path, headertype)

if not df is None:
    # Iris data: df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    st.write("### Current data frame:")
    st.write(df)

    dataset_name = st.text_input("Enter a name for your data set to be saved", value="")
    if dataset_name:
        newid = dataset_name + str(len(datasets) + 1)
        dataset = Dataset.from_dataframe(
            df, name=dataset_name, asset_path=dataset_name
        )
        # dataset.set_id(dataset_name + str(len(datasets)+1))
        # dataset.id= "sdhjshdjsdh"
        # st.write(dataset)
        # st.write("object above")
        # Reading the DataFrame back
        automl.registry.register(dataset)
        message = f"Dataset {dataset_name} has been succesfully stored and registered"
        st.success(message)
        retrieved_df = dataset.read()
        # DEBUG: st.write(retrieved_df)
