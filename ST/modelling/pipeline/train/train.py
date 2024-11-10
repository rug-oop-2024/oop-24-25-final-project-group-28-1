import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from sklearn.preprocessing import LabelEncoder

import ST.helpermods.helper as helper
from autoop.core.ml.model.classification import (
    SVM_classifier,
    logistic_regression,
    random_forest_classifier,
)
from autoop.core.ml.model.regression import (
    multiple_linear_regression,
    random_forest_regressor,
    ridge_regression,
)

# VOG: Shap and streamlit are not displaying well in Matplotlib context
# comment: import matplotlib
# comment: matplotlib.use("Agg")

# Developers info
# This is: modelling/pipeline/train/train.py
# Train the class and report the results of the pipeline.

st.write("""## Train and predict""")

dataset, automl = helper.get_selected_dataset()
if not dataset:
    st.stop()
    st.stop()

text = f"Selected dataset: {dataset.name}"
st.info(text)
# Debug: st.write(dataset.features.keys())
# This is a dictionary version of Feature objects, stored in a dataset objects
# in the database.

st.write("### User Input Parameters")
if not dataset.features.keys():
    st.warning("No features")
    st.stop()

if "selected_model" not in st.session_state:
    st.warning("No model selected")
    st.stop()


selected_model = st.session_state.selected_model

if not selected_model:
    st.warning("No model selected")
    st.stop()
else:
    st.write(f"Selected model is **{selected_model}**")


match selected_model:
    case "Logistic Regression":
        model = logistic_regression.LogisticRegressionModel(
            asset_path=""
        )
    case "Random forest classifier":
        model = random_forest_classifier.RandomForestClassifierModel(
            asset_path=""
        )
    case "SVM classifier":
        model = SVM_classifier.SVMClassifierModel(
            asset_path=""
        )
    case "Multiple linear regression":
        model = multiple_linear_regression.MultipleLinearRegression(
            asset_path=""
        )
    case "Random forest regressor":
        model = random_forest_regressor.RandomForestRegressorModel(
            asset_path=""
        )
    case "Ridge regression":
        model = ridge_regression.RidgeRegressionModel(
            asset_path=""
        )
    case _:
        st.write("Nothing selected")


def user_input_features():
    """Allow the user to input feature values by means of sliders"""
    data = {}
    selected_features = []
    categorical_column = None
    for key in dataset.features.keys():
        # comment: st.write("feat dict:", dataset.features[key])
        if dataset.features[key]["type"] == "numerical":
            minval = dataset.features[key]["minval"]
            maxval = dataset.features[key]["maxval"]
            av = (minval + maxval) / 2.0
            slider_val = st.slider(key, minval, maxval, av)
            data[key] = slider_val
            selected_features.append(key)
        else:
            categorical_column = key
    features = pd.DataFrame(data, index=[0])
    return features, selected_features, categorical_column


df_predict, selected_features, categorical_column = user_input_features()


st.write("### User Input parameters")
st.write(df_predict)

df = helper.dataset_to_pd(dataset)
df.columns = dataset.features.keys()


if categorical_column:
    X = df[selected_features]
    le = LabelEncoder()
    # Set makes names unique. List makes it subscriptable
    Ynames = list(set(df[categorical_column]))
    st.write("### The labels in the categorical column are:")
    st.write(Ynames)
    # Transform the Y values to integer numbers
    Y = le.fit_transform(df[categorical_column])
else:
    X = df[selected_features[:-1]]
    Y = selected_features[-1]
# Some checks for debugging
# debug: st.write("X:", X.shape)
# debug: st.write("Y:", Y)

model.fit(X, Y)

prediction = model.predict(df_predict)


st.write("### Class labels and their corresponding index number")
st.write(selected_features)

st.write("### Prediction")
st.write(Ynames[int(prediction)])
# debug: st.write(prediction)

if selected_model == "Random forest classifier":
    st.write("### Prediction Probability")
    prediction_proba = model.model.predict_proba(df_predict)
    st.write(prediction_proba)

    # Explaining the model's predictions using SHAP values
    # https://github.com/slundberg/shap

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    fig = (
        plt.gcf()
    )
    ax = fig.gca()
    fig.suptitle("Feature importance based on SHAP values", y=1.1)

    st.pyplot(fig)  # bbox_inches='tight')

# debug: explainer = shap.Explainer(clf, X)
# debug: shap_values = explainer(X, check_additivity=False)
# debug: shap.plots.bar(shap_values)
# debug: fig = plt.gcf()
# debug: st.pyplot(fig, bbox_inches='tight')

# Bar plot not working somehow. We keep the necessary code in these comments
# debug: plt.title('Feature importance based on SHAP values (Bar)')
# debug: shap.summary_plot(shap_values, X, plot_type="bar")
# debug: fig = plt.gcf()
# debug: st.pyplot(fig, bbox_inches='tight')
