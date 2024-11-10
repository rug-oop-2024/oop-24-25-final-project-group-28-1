from autoop.core.ml import metric
import ST.helpermods.helper as helper
from autoop.core.ml.model.classification import (logistic_regression, random_forest_classifier, SVM_classifier)
from autoop.core.ml.model.regression import (multiple_linear_regression, random_forest_regressor, ridge_regression)
from autoop.core.ml import (feature, pipeline)
import streamlit as st

# DEBUG: st.write("This is: modelling/pipeline/summary/summary.py")
# DEBUG: st.write("\n\n Prompt the user with a beautifuly formatted pipeline summary with all the configurations.")
# DEBUG: automl = AutoMLSystem.get_instance()

st.title("Pipeline Summary")
# pipeline = automl.get_pipeline()

# Get the selected dataset and automl instance
dataset, automl = helper.get_selected_dataset()
if not dataset:
    st.error("No dataset selected.")
    st.stop()

# Check if features are available in the dataset
if dataset.features is None:
    st.error("No features available in the dataset.")
    st.stop()

# Get the selected model from session state
if 'selected_model' not in st.session_state:
    st.warning("No model selected.")
    st.stop()

selected_model = st.session_state.selected_model
if not selected_model:
    st.warning("No model selected.")
    st.stop()
else:
    st.write(f"Selected model is **{selected_model}**")

# Match selected model to create the appropriate model instance
match selected_model:
    case "Logistic Regression":
        model = logistic_regression.LogisticRegressionModel()
    case "Random forest classifier":
        model = random_forest_classifier.RandomForestClassifierModel(name="model forest", asset_path="assets/models", model_type="classification")
    case "SVM classifier":
        model = SVM_classifier.SVMClassifierModel()
    case "Multiple linear regression":
        model = multiple_linear_regression.MultipleLinearRegression()
    case "Random forest regressor":
        model = random_forest_regressor.RandomForestRegressorModel()
    case "Ridge regression":
        model = ridge_regression.RidgeRegressionModel()
    case _:
        st.error("Invalid model selected.")
        st.stop()

# Load dataset into a DataFrame and ensure columns are named correctly
df = helper.dataset_to_pd(dataset)
if dataset.features:
    df.columns = dataset.features.keys()
else:
    st.error("Dataset does not contain features.")
    st.stop()

# Prepare features and target feature for the pipeline
features_list = []
target_feature = None
for name, feat_info in dataset.features.items():
    feat_type = feat_info.get("type", "numerical")
    feat = feature.Feature(name=name, type=feat_type)
    feat.calculate_statistics(df[name])
    if feat_type == "numerical":
        features_list.append(feat)
    else:
        target_feature = feat

# Initialize the pipeline
metrics = [metric.METRICS[0], metric.METRICS[1]]  # Assuming two metrics are needed
pipeline = pipeline.Pipeline(metrics, dataset, model, features_list, target_feature)

# Display pipeline summary
if pipeline:
    st.write("### Pipeline Overview")

    # Model configuration
    st.subheader("Model Configuration")
    st.write(f"**Model Type**: {model.model_type}")
    st.write(f"**Model Name**: {model.__class__.__name__}")
    if hasattr(model, "parameters") and model.parameters:
        st.write("**Parameters:**")
        for param, value in model.parameters.items():
            st.write(f" - {param}: {value}")

    # Dataset configuration
    st.subheader("Dataset Configuration")
    st.write(f"**Dataset Name**: {dataset.name}")
    st.write(f"**Asset Path**: {dataset.asset_path}")
    st.write(f"**Number of Samples**: {len(df)}")

    # Feature configuration
    st.subheader("Features")
    st.write("**Input Features:**")
    for feat in pipeline._input_features:
        st.write(f" - {feat.name} ({feat.type})")
    if target_feature:
        st.write(f"**Target Feature**: {target_feature.name} ({target_feature.type})")

    # Metrics configuration
    st.subheader("Metrics")
    if pipeline._metrics:
        for met in pipeline._metrics:
            st.write(f" - {met.__class__.__name__}")
    else:
        st.write("No metrics configured.")

    # Data-split configuration
    st.subheader("Data Split")
    st.write(f"**Training Split**: {pipeline._split * 100}%")
    st.write(f"**Testing Split**: {(1 - pipeline._split) * 100}%")

    # Artifacts summary
    st.subheader("Artifacts")
    artifacts = pipeline.artifacts
    if artifacts:
        for artifact in artifacts:
            st.write(f" - {artifact.name}")
    else:
        st.write("No artifacts generated.")
else:
    st.error("No pipeline configuration found. Please create and configure a pipeline first.")
