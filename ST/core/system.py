from pathlib import Path
from typing import List
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import LocalStorage, Storage


class ArtifactRegistry:
    """class to register artifacts"""
    def __init__(self, database: Database, storage: Storage):
        """Initialize ArtifactRegistry class with a database and storage"""
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact):
        """Register the artifact"""
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """Make a list of the artifacts"""
        entries = self._database.list("artifacts")
        artifacts = []
        for _id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """Getter for artifact"""
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str):
        """delete an artifact"""
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """class for AutoMLSystem. Used to set up database and file storage.
    Can read artifacts from a list of artifacts"""
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        """initialize AutoMLSystem with storage and database"""
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance():
        """getter for an instance of AutoMLSystem"""
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage(Path("./assets/objects")),
                Database(LocalStorage(Path("./assets/dbo"))),
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self):
        """property: registry"""
        return self._registry
