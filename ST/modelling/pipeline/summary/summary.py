from autoop.core.ml import metric
import ST.helpermods.helper as helper
from autoop.core.ml.model.classification import (logistic_regression, random_forest_classifier, SVM_classifier)
from autoop.core.ml.model.regression import (multiple_linear_regression, random_forest_regressor, ridge_regression)
from autoop.core.ml import (feature, pipeline)

import streamlit as st

#DEBUG: st.write("This is: modelling/pipeline/summary/summary.py")
#DEBUG: st.write("\n\n Prompt the user with a beautifuly formatted pipeline summary with all the configurations.")



# DEBUG: automl = AutoMLSystem.get_instance()

st.title("Pipeline Summary")
# pipeline = automl.get_pipeline()

metric = [metric.METRICS[0], metric.METRICS[1]]
dataset, automl = helper.get_selected_dataset()
if not dataset:
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
    case "Logistic Regression":
        model = logistic_regression.LogisticRegressionModel(asset_path="")
    case "Random forest classifier":
        model = random_forest_classifier.RandomForestClassifierModel(asset_path="")
    case "SVM classifier":
        model = SVM_classifier.SVMClassifierModel(asset_path="")
    case "Multiple_linear_regression":
        model = multiple_linear_regression.MultipleLinearRegression(asset_path="")        
    case "Random_forest_regressor":
        model = random_forest_regressor.RandomForestRegressorModel(asset_path="")
    case "Ridge regression":
        model = ridge_regression.RidgeRegressionModel(asset_path="")
    case _:
        st.write("Noting selected")

df = helper.dataset_to_pd(dataset)
num_cols = len(df.columns)
dataset_features = dataset.get_features()
df.columns = dataset.features.keys()

features_list = []
target_feature = None
for name in dataset_features.keys():
    feat_type = dataset_features[name]['type']
    if feat_type == 'numerical':
        feat = feature.Feature(name=name, type=feat_type)
        feat.calculate_statistics(df[name])
        features_list.append(feat)
    else:
        target_feature = feature.Feature(name=name, type=feat_type)
        target_feature.calculate_statistics(df[name])

if 'split_val' not in st.session_state:
    st.warning("No split value selected")
    st.stop()
else:
    split_val = st.session_state.split_val
    if split_val is None:
        split_val = 0.8
        
pipeline = pipeline.Pipeline(metric, dataset, model, features_list, target_feature, split=split_val)


if pipeline:
    st.write("### Pipeline Overview")

    # model configuration
    st.subheader("Model Configuration")
    model = pipeline.model
    st.write(f"**Model Type**: {model.type}")
    st.write(f"**Model Name**: {model.__class__.__name__}")
    if hasattr(model, 'parameters') and model.parameters:
        st.write("**Parameters:**")
        for param, value in model.parameters.items():
            st.write(f" - {param}: {value}")

    # dataset configuration
    st.subheader("Dataset Configuration")
    dataset = pipeline._dataset
    st.write(f"**Dataset Name**: {dataset.name}")
    st.write(f"**Asset Path**: {dataset.asset_path}")
    st.write(f"**Number of Samples**: {dataset.size}")

    # feature configuration
    st.subheader("Features")
    st.write("**Input Features:**")
    for feature in pipeline._input_features:
        st.write(f" - {feature.name} ({feature.type})")

    st.write(f"""**Target Feature**: {pipeline._target_feature.name}
             ({pipeline._target_feature.type})""")

    # metrics configuration
    st.subheader("Metrics")
    if pipeline._metrics:
        for metric in pipeline._metrics:
            st.write(f" - {metric.__class__.__name__}")
    else:
        st.write("No metrics configured.")

    # data-split configuration
    st.subheader("Data Split")
    st.write(f"**Training Split**: {pipeline._split * 100}%")
    st.write(f"**Testing Split**: {(1 - pipeline._split) * 100}%")

    # artifacts summary
    st.subheader("Artifacts")
    artifacts = pipeline.artifacts
    if artifacts:
        for artifact in artifacts:
            st.write(f" - {artifact.name}")
    else:
        st.write("No artifacts generated.")

else:
    st.error("""No pipeline configuration found. Please create and configure a
             pipeline first.""")

