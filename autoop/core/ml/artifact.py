import base64
from typing import Any, Dict


class Artifact:
    def __init__(
        self,
        name: str,
        tags: str,
        type: str,
        asset_path: str,
        data: bytes,
        version: str = "1.0.0",
        metadata: Dict[str, Any] = None,
        features: Dict[str, Any] = None
    ):
        self.name = name
        self.tags = tags
        self.type = type
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.metadata = metadata or {}
        self.features = features

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
