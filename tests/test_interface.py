import pytest

from predict_ur_score.exceptions import PredictorInvalidInput
from predict_ur_score.utils.interface import Interface


def test_invalid_home_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Cat")

    interface = Interface()

    with pytest.raises(PredictorInvalidInput):
        interface._check_home_input()


def test_invalid_gender(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Cat")

    interface = Interface()

    with pytest.raises(PredictorInvalidInput):
        interface._check_gender()


def test_invalid_part_time_job(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")

    interface = Interface()

    with pytest.raises(PredictorInvalidInput):
        interface._check_part_time_job()


def test_invalid_absence_negative(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "-1")

    interface = Interface()

    with pytest.raises(PredictorInvalidInput):
        interface._check_absence_days()


def test_invalid_extra_activity(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "-1")

    interface = Interface()

    with pytest.raises(PredictorInvalidInput):
        interface._check_extracurricular_activities()


def test_invalid_absence_str(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Cat")

    interface = Interface()

    with pytest.raises(PredictorInvalidInput):
        interface._check_absence_days()


def test_invalid_weekly_self_study_hours_negative(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "-1")

    interface = Interface()

    with pytest.raises(PredictorInvalidInput):
        interface._check_weekly_self_study_hours()


def test_invalid_weekly_self_study_hours_str(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Cat")

    interface = Interface()

    with pytest.raises(PredictorInvalidInput):
        interface._check_weekly_self_study_hours()


def test_invalid_career_aspiration(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Unknown")

    interface = Interface()

    with pytest.raises(PredictorInvalidInput):
        interface._check_career_aspiration()


def test_stat_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "Cat")

    interface = Interface()

    with pytest.raises(PredictorInvalidInput):
        interface._check_stat_input()
