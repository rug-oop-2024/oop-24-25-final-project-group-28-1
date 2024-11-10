import streamlit as st

st.write("# AutOOP AutoML library")

line1 = """This application that allows you to train models by uploading a CSV and 
choosing the target column.
"""

line2 = """* Upload a CSV dataset (e.g., Iris) and convert that into a dataset
* Save the converted dataset
* Delete existing datasets
* Load stored datasets
* Detect and select features
* Select a model
* Model a pipeline
* Specify a data split to select training data
* Select metrics
* Print a pipeline summary
* Train  and report the results of the pipeline
* Save a pipeline
* Load an existing pipeline
* Predict results based on a CSV file
"""

st.write("## Description of the application")
st.write(line1)
st.write("## Features")
st.write(line2)

st.write(f"\n\n\n Streamlit version: {st.__version__}") 
