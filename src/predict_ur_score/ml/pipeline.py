from abc import ABC, abstractmethod

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from predict_ur_score.config import RANDOM_STATE

from .preprocessing import DataPreprocessor


class BaseModelPipeline(ABC):
    """A class to create a base pipeline for predicting student scores."""

    def __init__(self) -> None:
        self.preprocessor = DataPreprocessor().transformer_pipeline()

    @abstractmethod
    def create_model(self) -> RegressorMixin:
        """Abstract method to create a machine learning model."""

    def create_pipeline(self) -> Pipeline:
        """Creates a machine learning pipeline with preprocessing and model."""

        model = self.create_model()
        pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("model", model)]
        )

        return pipeline


class RandomForestPipeline(BaseModelPipeline):
    """A class to create a Random Forest pipeline for predicting student scores."""

    def create_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        )


class DecisionTreePipeline(BaseModelPipeline):
    """A class to create a Decision Tree pipeline for predicting student scores."""

    def create_model(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(random_state=RANDOM_STATE)


class SVMPipeline(BaseModelPipeline):
    """A class to create a Support Vector Machine (SVM) pipeline for predicting student scores."""

    def create_model(self) -> MultiOutputRegressor:
        return MultiOutputRegressor(SVR(kernel="rbf"))


class LinearRegressionPipeline(BaseModelPipeline):
    """A class to create a Linear Regression pipeline for predicting student scores."""

    def create_model(self) -> LinearRegression:
        return LinearRegression()
