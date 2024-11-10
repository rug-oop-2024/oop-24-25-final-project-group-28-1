import ST.helpermods.helper as helper
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt

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

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df_predict)
prediction_proba = clf.predict_proba(df_predict)

st.write('### Class labels and their corresponding index number')
st.write(selected_features)

st.write('### Prediction')
st.write(Ynames[int(prediction)])
#st.write(prediction)

st.write('### Prediction Probability')
st.write(prediction_proba)

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
fig = plt.gcf()    # We need a figure in st.pyplot because otherwise streamlit complains with warnings
st.pyplot(fig, bbox_inches='tight')

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

