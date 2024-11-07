from pydantic import BaseModel, Field
from typing import Literal
import numpy as np


class Feature(BaseModel):
    name: str = Field(..., description="Name of the feature")
    feature_type: Literal["categorical", "numerical"] = Field(
        description="Type of the feature"
        )
    unique_values: int = Field(
        None,
        description="Number of unique values for categorical features"
        )
    mean: float = Field(
        None,
        description="Mean value for numerical features"
        )
    std_dev: float = Field(
        None,
        description="Standard deviation for numerical features"
        )

    def calculate_statistics(self, data: np.ndarray) -> None:
        """
        Calculates and sets statistics based on feature type.
        Args:
            data (np.ndarray): Data for the feature column.
        """
        if self.feature_type == "numerical":
            self.mean = np.mean(data)
            self.std_dev = np.std(data)
        elif self.feature_type == "categorical":
            self.unique_values = len(np.unique(data))

    def __str__(self) -> str:
        """
        Custom string representation of the Feature instance.
        Returns:
            str: Information about the feature.
        """
        if self.feature_type == "numerical":
            return (
                f"Feature(name={self.name}, type={self.feature_type}, "
                f"mean={self.mean:.2f}, std_dev={self.std_dev:.2f})"
            )
        elif self.feature_type == "categorical":
            return (
                f"Feature(name={self.name}, type={self.feature_type}, "
                f"unique_values={self.unique_values})"
            )
        return f"Feature(name={self.name}, type={self.feature_type})"
