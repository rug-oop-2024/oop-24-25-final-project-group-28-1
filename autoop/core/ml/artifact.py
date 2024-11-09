import base64
from typing import Any, Dict, List
from autoop.core.ml.feature import Feature
from pydantic import BaseModel, Field


class Artifact(BaseModel):
    name: str = Field(description="Name of asset")
    tags: str = Field(default="No tags", description="Tags")
    type: str = Field(description="The type of the artifact")
    asset_path: str = Field(description="Path where the artifact is stored")
    version: str = Field(default="1.0.0", description="Version of the artifact")
    data: bytes = Field(description="Binary data of the artifact")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata for the artifact"
    )
    features:  Dict[str, Any] = Field(default={}, description="List with features")

    @property
    def id(self) -> str:
        """Generate a unique ID based on asset_path and version."""
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        # return f"{encoded_path}:{self.version}"
        return f"{encoded_path}_{self.version}"

    def save(self, data: bytes) -> bytes:
        """
        Saves the artifact data.
        Args:
            data (bytes): Data to save as part of the artifact.
        Returns:
            bytes: The saved data.
        """
        self.data = data
        return self.data

    def read(self) -> bytes:
        """
        Reads the artifact data.
        Returns:
            bytes: The data of the artifact.
        """
        return self.data
    
    
    def get_features(self):
        return self.features
    
    def add_features(self, features: Dict[str, Any]) -> None:
        self.features = features
