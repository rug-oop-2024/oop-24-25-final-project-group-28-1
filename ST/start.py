import streamlit as st

page_manual = st.Page("page/docs/manual.py", title="Manual")
page_datasets_info = st.Page("page/datasets/datasets_info.py", title="Dataset management")
page_datasets_create = st.Page("datasets/management/create/create_dataset.py", title="Create dataset")
page_datasets_save = st.Page("datasets/management/save/save_dataset.py", title="Save dataset")
page_datasets_delete = st.Page("datasets/management/delete/delete_dataset.py", title="Delete dataset")
page_modelling_info = st.Page("page/modelling/modelling_info.py", title="Modelling")
page_modelling_load = st.Page("modelling/datasets/list/load_dataset.py", title="Load exiting data sets")
page_modelling_feat = st.Page("modelling/datasets/features/dataset_features.py", title="Detect features")
page_modelling_models = st.Page("modelling/models/models.py", title="Select model" )
page_modelling_pipeline_split = st.Page("modelling/pipeline/split/split.py", title="Select a dataset split")
page_modelling_pipeline_metrics = st.Page("modelling/pipeline/metrics/metrics.py", title="Select metrics")
page_modelling_pipeline_summary = st.Page("modelling/pipeline/summary/summary.py", title="Pipeline summary")
page_modelling_pipeline_train = st.Page("modelling/pipeline/train/train.py", title="Train the class")
page_modelling_pipeline_save = st.Page("modelling/pipeline/save/save_pipeline.py", title="Save pipeline")
page_deployment_overview = st.Page("page/deployment/overview.py", title="Existing saved pipelines")
page_deployment_load = st.Page("deployment/load/deployment_load.py", title="Select pipeline")
page_deployment_predict = st.Page("deployment/predict/deployment_predict.py", title="Predict given a CSV file")

pages = {
    "MANUAL":    [page_manual],
    "DATASETS":  [page_datasets_create, page_datasets_delete],
    "MODELLING": [page_modelling_load, page_modelling_feat, page_modelling_models,
                  page_modelling_pipeline_split, page_modelling_pipeline_metrics, page_modelling_pipeline_summary,
                  page_modelling_pipeline_train, page_modelling_pipeline_save ],
    "DEPLOYMENT":[page_deployment_load, page_deployment_predict],
}

pg = st.navigation(pages)
pg.run()


#ST/page/datasets: Create a page where you can manage the datasets.
#ST/datasets/management/create: Upload a CSV dataset (e.g., Iris) and convert that into a dataset using the from_dataframe factory method. 
#Since a dataset is already an artifact, you can use the AutoMLSystem.get_instance singelton class to to access either storage, 
#database, or the artifact registry to save it.
#ST/datasets/management/save: Use the artifact registry to save converted dataset artifact object.
#ST/page/modelling: Create a page where you will be modelling a pipeline.
#ST/modelling/datasets/list: Load existing datasets using the artifact registry. You can use a select box to achieve this.
#ST/modelling/datasets/features: Detect the features and generate a selection menu for selecting the input features (many) and one target feature. Based on the feature selections, prompt the user with the detected task type (i.e., classification or regression).
#ST/modelling/models: Prompt the user to select a model based on the task type.
#ST/modelling/pipeline/split: Prompt the user to select a dataset split.
#ST/modelling/pipeline/metrics: Prompt the user to select a set of compatible metrics.
#ST/modelling/pipeline/summary: Prompt the user with a beautifuly formatted pipeline summary with all the configurations.
#ST/modelling/pipeline/train: Train the class and report the results of the pipeline.
#ST/modelling/pipeline/save: Prompt the user to give a name and version for the pipeline and convert it into an artifact which can be saved.

#    ST/page/deployment: Create a page where you can see existing saved pipelines.
#    ST/deployment/load: Allow the user to select existing pipelines and based on the selection show a pipeline summary.
#    ST/deployment/predict: Once the user loads a pipeline, prompt them to provide a CSV on which they can perform predictions.

