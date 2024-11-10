import streamlit as st
from ST.core.system import AutoMLSystem

# Get instance of AutoMLSystem
automl = AutoMLSystem.get_instance()

# Retrieve the pipeline artifact from the registry if it exists
st.title("Pipeline Summary")
pipeline_artifacts = automl.registry.list(type="pipeline")

if pipeline_artifacts:
    # assume only one pipeline artifact
    pipeline = pipeline_artifacts[0]
    st.write("### Pipeline Overview")

    # Model configuration
    st.subheader("Model Configuration")
    model = pipeline.model
    st.write(f"**Model Type**: {model.model_kind}")
    st.write(f"**Model Name**: {model.__class__.__name__}")
    if hasattr(model, 'parameters') and model.parameters:
        st.write("**Parameters:**")
        for param, value in model.parameters.items():
            st.write(f" - {param}: {value}")

    # Dataset configuration
    st.subheader("Dataset Configuration")
    dataset = pipeline._dataset
    st.write(f"**Dataset Name**: {dataset.name}")
    st.write(f"**Asset Path**: {dataset.asset_path}")
    st.write(f"**Number of Samples**: {dataset.size}")

    # Feature configuration
    st.subheader("Features")
    st.write("**Input Features:**")
    for feature in pipeline._input_features:
        st.write(f" - {feature.name} ({feature.type})")

    st.write(f"""**Target Feature**: {pipeline._target_feature.name}
             ({pipeline._target_feature.type})""")

    # Metrics configuration
    st.subheader("Metrics")
    if pipeline._metrics:
        for metric in pipeline._metrics:
            st.write(f" - {metric.__class__.__name__}")
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
    st.error("""No pipeline configuration found. Please create and
             configure a pipeline first.""")
