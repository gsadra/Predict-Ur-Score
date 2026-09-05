from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from .pipeline import (
    DecisionTreePipeline,
    LinearRegressionPipeline,
    RandomForestPipeline,
    SVMPipeline,
)
from predict_ur_score.exceptions import PredictorInvalidModel
from predict_ur_score.utils import load_data


class ModelTrainer:
    '''A class to train machine learning models for predicting student scores.'''

    MODEL_MAPPING = {
        'random_forest': RandomForestPipeline,
        'decision_tree': DecisionTreePipeline,
        'svm': SVMPipeline,
        'linear_regression': LinearRegressionPipeline
    }

    TARGETS = [
        'math_score',
        'history_score',
        'physics_score',
        'chemistry_score',
        'biology_score',
        'english_score',
        'geography_score',
    ] 

    def __init__(self, model_type: str = 'random_forest', model_path: Path | None = None) -> None:
        self.data = load_data()
        self.model_type: str = model_type
        self.model_path: Path = model_path
        self.pipeline: Pipeline = self._select_pipeline()

    def train_model(self) -> Pipeline:
        X, y = self._load_training_data()
        self.pipeline.fit(X, y)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)

        print(f'Model trained and saved at: {self.model_path}')

    
    def _load_training_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        X = self.data.drop(columns=self.TARGETS)
        y = self.data[self.TARGETS]

        return X, y

    def _select_pipeline(self) -> Pipeline:
        if self.model_type not in self.MODEL_MAPPING:
            raise PredictorInvalidModel(f'Invalid model type: {self.model_type}. '
                                       f'Choose from {list(self.MODEL_MAPPING.keys())}')

        pipeline_class = self.MODEL_MAPPING[self.model_type]
        return pipeline_class().create_pipeline()

if __name__ == "__main__":
    model_trainer = ModelTrainer(model_type='random_forest', model_path=Path('models/random_forest_model.joblib'))
    model_trainer.train_model()