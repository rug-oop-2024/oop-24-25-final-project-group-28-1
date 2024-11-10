from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """
    Pipeline class for training and evaluating a model with given features.
    """
    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split=0.8,
    ):
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        self._predictions = None
        self._metric_results = []

        if (
            target_feature.type == "categorical"
            and model._model_type != "classification"
        ):
            raise ValueError(
                """Model type must be classification for categorical
                target feature""")

        if (
            target_feature.type == "continuous"
        ) and (
            model._model_type != "regression"
        ):
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self):
        return (
            f"Pipeline(\n"
            f"    model={self._model._model_type},\n"
            f"    input_features={list(map(str, self._input_features))},\n"
            f"    target_feature={str(self._target_feature)},\n"
            f"    split={self._split},\n"
            f"    metrics={list(map(str, self._metrics))},\n"
            f")"
        )

    @property
    def model(self):
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Get the artifacts generated during the pipeline execution to be saved.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = pickle.dumps(artifact["encoder"])
                artifacts.append(Artifact(name=name, data=data))
            elif artifact_type in ["StandardScaler"]:
                data = pickle.dumps(artifact["scaler"])
                artifacts.append(Artifact(name=name, data=data))

        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(
                name=self._model.name,
                tags="No Tag",
                type=self._model._model_type,
                asset_path="",
                data=pickle.dumps(pipeline_data)
            )
        )
        artifacts.append(
            self._model.to_artifact(
                name=f"pipeline_model_{self._model._model_type}"
            )
        )
        return artifacts

    def _register_artifact(self, name: str, artifact):
        self._artifacts[name] = artifact

    def _preprocess_features(self):
        target_data = preprocess_features(
            [self._target_feature], self._dataset)[0]
        target_feature_name, target_data, artifact = target_data
        self._register_artifact(target_feature_name, artifact)

        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)

        self._output_vector = target_data
        self._input_vectors = [data for _, data, _ in input_results]

    def _split_data(self):
        split = int(self._split * len(self._output_vector))
        self._train_X = [vector[:split] for vector in self._input_vectors]
        self._test_X = [vector[split:] for vector in self._input_vectors]
        self._train_y = self._output_vector[:split]
        self._test_y = self._output_vector[split:]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        return np.concatenate(vectors, axis=1)

    def _train(self):
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self, X=None, Y=None):

        # Use default test data if X and Y are not provided
        if X is None:
            X = self._compact_vectors(self._test_X)
        if Y is None:
            Y = self._test_y

        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric(predictions, Y)
            self._metric_results.append((metric, result))
        self._predictions = predictions
        return self._metric_results

    def execute(self):
        self._preprocess_features()
        self._split_data()
        self._train()

        train_X = self._compact_vectors(self._train_X)
        test_X = self._compact_vectors(self._test_X)

        train_metrics_results = self._evaluate(train_X, self._train_y)
        test_metrics_results = self._evaluate(test_X, self._test_y)

        return {
            "train_metrics": train_metrics_results,
            "test_metrics": test_metrics_results,
            "predictions": self._model.predict(test_X),
        }
