from pydantic import BaseModel, Field
import pandas as pd
import io

# Define the Artifact base class
class Artifact(BaseModel):
    type: str = Field(..., description="The type of the artifact")
    data: bytes = Field(..., description="The byte-encoded data")

    def read(self) -> bytes:
        """Method to retrieve data (to be overridden by subclasses if necessary)."""
        return self.data
    
    def save(self, data: bytes) -> None:
        """Method to save data (to be overridden by subclasses if necessary)."""
        self.data = data

# Define the Dataset class inheriting from Artifact
class Dataset(Artifact):

    def __init__(self, *args, **kwargs):
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str, version: str="1.0.0"):
        # Encode DataFrame as CSV bytes
        encoded_data = data.to_csv(index=False).encode()
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=encoded_data,
            version=version,
        )
        
    def read(self) -> pd.DataFrame:
        # Decode bytes back to CSV and load into a DataFrame
        csv = self.data.decode()
        return pd.read_csv(io.StringIO(csv))
    
    def save(self, data: pd.DataFrame) -> None:
        # Convert DataFrame to CSV bytes and save
        encoded_data = data.to_csv(index=False).encode()
        super().save(encoded_data)

# Example usage
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
dataset = Dataset.from_dataframe(df, name="example_dataset", asset_path="/path/to/dataset")

# Reading the DataFrame back
retrieved_df = dataset.read()
print(retrieved_df)

