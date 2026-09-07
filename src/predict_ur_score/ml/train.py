from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from predict_ur_score.config import BEST_MODEL_PATH
from predict_ur_score.exceptions import PredictorInvalidModel
from predict_ur_score.utils import load_data

from .constants import BEST_MODEL_PARAMS, TARGETS
from .pipeline import (
    DecisionTreePipeline,
    LinearRegressionPipeline,
    RandomForestPipeline,
    SVMPipeline,
)


class ModelTrainer:
    """A class to train machine learning models for predicting student scores."""

    MODEL_MAPPING = {
        "random_forest": RandomForestPipeline,
        "decision_tree": DecisionTreePipeline,
        "svm": SVMPipeline,
        "linear_regression": LinearRegressionPipeline,
    }

    def __init__(
        self,
        model_type: str = "random_forest",
        model_path: Path = BEST_MODEL_PATH,
    ) -> None:
        self.data = load_data()
        self.model_type: str = model_type
        self.model_path: Path | None = model_path
        self.pipeline: Pipeline = self._select_pipeline()

    def train_model(self) -> None:
        X, y = self._load_training_data()
        self.pipeline.fit(X, y)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)

        return self.pipeline

    def _load_training_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        X = self.data.drop(columns=TARGETS)
        y = self.data[TARGETS]

        return X, y

    def _select_pipeline(self) -> Pipeline:
        if self.model_type not in self.MODEL_MAPPING:
            raise PredictorInvalidModel(
                f"Invalid model type: {self.model_type}. "
                f"Choose from {list(self.MODEL_MAPPING.keys())}"
            )

        pipeline_class = self.MODEL_MAPPING[self.model_type]
        pipeline = pipeline_class().create_pipeline()

        self._set_best_params(pipeline)

        return pipeline

    def _set_best_params(self, pipeline: Pipeline) -> None:
        params = BEST_MODEL_PARAMS.get(self.model_type)

        if params:
            pipeline.set_params(**params)


if __name__ == "__main__":
    model_trainer = ModelTrainer(
        model_type="random_forest",
        model_path=Path("models/best_model.joblib"),
    )
    model_trainer.train_model()
