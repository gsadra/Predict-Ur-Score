import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predict_ur_score.utils import load_data


class DataPreprocessor:
    """A class to preprocess the student scores data for machine learning tasks."""

    def __init__(self) -> None:
        self.df: pd.DataFrame = load_data()
        self.ohe: OneHotEncoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.scaler: StandardScaler = StandardScaler()

    def transformer_pipeline(self) -> ColumnTransformer:
        categorical_features = self.df.select_dtypes(
            include=["str"]
        ).columns
        numerical_features = ["absence_days", "weekly_self_study_hours"]

        numerical_pipeline = Pipeline(steps=[("scaler", self.scaler)])

        categorical_pipeline = Pipeline(steps=[("onehot", self.ohe)])

        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numerical_pipeline, numerical_features),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features,
                ),
            ]
        )

        return preprocessor
