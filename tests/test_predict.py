import pytest

from predict_ur_score.ml.predict import Predictor


@pytest.fixture
def predictor() -> Predictor:
    return Predictor()


@pytest.fixture
def student_sample():
    return {
        "gender": "female",
        "part_time_job": False,
        "absence_days": 5,
        "extracurricular_activities": True,
        "weekly_self_study_hours": 15,
        "career_aspiration": "Unknown",
    }


def test_predictor_loads_model(predictor: Predictor):
    assert predictor.model is not None


def test_prediction_returns_targets(
    predictor: Predictor, student_sample: dict
):
    predictions = predictor.predict(student_sample)

    expected_targets = {
        "math_score",
        "history_score",
        "physics_score",
        "chemistry_score",
        "biology_score",
        "english_score",
        "geography_score",
    }

    assert set(predictions.keys()) == expected_targets
