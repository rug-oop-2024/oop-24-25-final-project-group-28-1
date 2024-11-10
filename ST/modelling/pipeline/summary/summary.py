import streamlit as st
from ST.core.system import AutoMLSystem


automl = AutoMLSystem.get_instance()

st.title("Pipeline Summary")
pipeline = automl.get_pipeline()

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
