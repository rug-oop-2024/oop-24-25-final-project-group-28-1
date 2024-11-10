from autoop.core.ml.model.classification import logistic_regression
from autoop.core.ml.model.classification import random_forest_classifier
from autoop.core.ml.model.classification import SVM_classifier
from autoop.core.ml.model.regression import multiple_linear_regression
from autoop.core.ml.model.regression import random_forest_regressor
from autoop.core.ml.model.regression import ridge_regression

import ST.helpermods.helper as helper
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# VOG: Shap and streamlit are not displaying well in Matplotlib context
#import matplotlib
#matplotlib.use("Agg")

#st.write("**Developers info**")
#st.write("This is: modelling/pipeline/train/train.py")
#st.write("\n\n Train the class and report the results of the pipeline.")
#st.write("---")

st.write("""## Train and predict""")

dataset, automl = helper.get_selected_dataset()
if not dataset:
     st.stop()

s = f"Selected dataset: {dataset.name}"
st.info(s)
# Debug: st.write(dataset.features.keys())
# This is a dictionary version of Feature objects, stored in a dataset objects
# in the database.

st.write('### User Input Parameters')
if not dataset.features.keys():
    st.warning("No features")
    st.stop()

if 'selected_model' not in st.session_state:
    st.warning("No model selected")
    st.stop()
   
selected_model = st.session_state.selected_model

if not selected_model:    
    st.warning("No model selected")
    st.stop()
else:
    st.write(f"Selected model is **{selected_model}**")
    
match selected_model:
    case "Logistic Regresssion":
        model = logistic_regression.LogisticRegressionModel()
    case "Random forest classifier":
        model = random_forest_classifier.RandomForestClassifierModel('classification')
    case "SVM classifier":
        model = SVM_classifier.SVMClassifierModel()
    case "Multiple_linear_regression":
        model = multiple_linear_regression.MultipleLinearRegression()        
    case "Random_forest_regressor":
        model = random_forest_regressor.RandomForestRegressorModel()
    case "Ridge regression":
        model = ridge_regression.RidgeRegressionModel()
    case _:
        st.write("Noting selected")

    
def user_input_features():
    data = {}
    selected_features = []
    for key in dataset.features.keys():
        # st.write("feat dict:", dataset.features[key])
        if dataset.features[key]['type'] == 'numerical':
            minval = dataset.features[key]['minval']
            maxval = dataset.features[key]['maxval']
            av = (minval+maxval)/2.0
            slider_val = st.slider(key, minval, maxval, av)
            data[key] = slider_val
            selected_features.append(key)
        else:
            categorical_column = key
    features = pd.DataFrame(data, index=[0])
    return features, selected_features, categorical_column

df_predict, selected_features, categorical_column = user_input_features()

st.write('### User Input parameters')
st.write(df_predict)

df = helper.dataset_to_pd(dataset)
df.columns = dataset.features.keys()

X = df[selected_features]
le = LabelEncoder()
Ynames = list(set(df[categorical_column]))  # Set makes names unique. List makes it subscriptable
st.write("### The labels in the categorical column are:")
st.write(Ynames)
Y = le.fit_transform(df[categorical_column])  # Transform the Y values to integer numbers

# Some checks for debugging
#st.write("X:", X.shape)
#st.write("Y:", Y)

#clf = RandomForestClassifier()
model.fit(X, Y)

prediction = model.predict(df_predict)
prediction_proba = model.model.predict_proba(df_predict)

st.write('### Class labels and their corresponding index number')
st.write(selected_features)

st.write('### Prediction')
st.write(Ynames[int(prediction)])
#st.write(prediction)

st.write('### Prediction Probability')
st.write(prediction_proba)

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap

explainer = shap.TreeExplainer(model.model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
fig = plt.gcf()    # We need a figure in st.pyplot because otherwise streamlit complains with warnings
ax = fig.gca()
fig.suptitle('Feature importance based on SHAP values', y=1.1)

st.pyplot(fig)  #, bbox_inches='tight')

#explainer = shap.Explainer(clf, X) 
#shap_values = explainer(X, check_additivity=False)
#shap.plots.bar(shap_values)
#fig = plt.gcf()    # We need a figure in st.pyplot because otherwise streamlit complains with warnings
#st.pyplot(fig, bbox_inches='tight')

# Bar plot not working somehow. We keep the necessary code in these comments
#plt.title('Feature importance based on SHAP values (Bar)')
#shap.summary_plot(shap_values, X, plot_type="bar")
#fig = plt.gcf()
#st.pyplot(fig, bbox_inches='tight')

