import joblib
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from predict_ur_score.exceptions import (
    PredictorInvalidModel, 
    PredictorFileNotFound,
    PredictorMissingFeatures
)
from predict_ur_score.config import BEST_MODEL_PATH
from .constants import TARGETS, FEATURES


class Predictor:
    '''Predict each score based on inputs.'''

    def __init__(self, model_path: Path = BEST_MODEL_PATH) -> None:
        self.model_path = model_path
        self.model = self._load_model()

    def predict(self, student_data: dict) -> dict[str, float]:
        input_data = self._collect_input_data(student_data)
        prediction = self.model.predict(input_data)[0]

        return dict(zip(TARGETS, prediction))

    def _load_model(self) -> Pipeline:
        if not self.model_path.exists():
            raise PredictorFileNotFound(f'Model file not found in {self.model_path}')

        model = joblib.load(self.model_path)

        if not isinstance(model, Pipeline):
            raise PredictorInvalidModel('The loaded model is not a valid sklearn pipline.')

        return model

    def _collect_input_data(self, student_data: dict) -> pd.DataFrame:
        missing_features = set(FEATURES) - set(student_data)

        if missing_features:
            raise PredictorMissingFeatures(f'Missing features: {missing_features}')

        return pd.DataFrame([student_data], columns=FEATURES)


if __name__ == '__main__':
    predictor = Predictor()
    student = {
        'gender': 'male',
        'part_time_job': False,
        'absence_days': 0,
        'extracurricular_activities': True,
        'weekly_self_study_hours': 30,
        'career_aspiration': 'Lawyer'
        }
    
    predictions = predictor.predict(student)
    print(predictions)
