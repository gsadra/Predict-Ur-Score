from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from predict_ur_score.config import RANDOM_STATE
from predict_ur_score.utils import load_data

from .constants import TARGETS
from .pipeline import (
    DecisionTreePipeline,
    LinearRegressionPipeline,
    RandomForestPipeline,
    SVMPipeline,
)


@dataclass
class ModelSearch:
    model_name: str
    best_params: dict
    best_rmse: float
    best_estimator: Pipeline


class ModelValidation:
    """Evaluate regression models using cross-validation"""

    SEARCH_SPACES: dict = {
        "random_forest": {
            "pipeline": RandomForestPipeline,
            "params": {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 1.0],
            },
        },
        "decision_tree": {
            "pipeline": DecisionTreePipeline,
            "params": {
                "model__max_depth": [None, 5, 10, 15, 20, 30],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": [None, "sqrt", "log2"],
            },
        },
        "svm": {
            "pipeline": SVMPipeline,
            "params": {
                "model__estimator__C": [0.1, 1, 10, 100],
                "model__estimator__gamma": ["scale", "auto", 0.01, 0.1],
                "model__estimator__epsilon": [
                    0.01,
                    0.1,
                    0.2,
                    0.5,
                ],
            },
        },
        "linear_regression": {
            "pipeline": LinearRegressionPipeline,
            "params": {},
        },
    }

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = RANDOM_STATE,
        n_iter: int = 50,
    ) -> None:
        self.cv = KFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        self.random_state = random_state
        self.n_iter = n_iter

    def tune_model(
        self, model_name: str, X: pd.DataFrame, y: pd.DataFrame
    ) -> ModelSearch:
        configuration = self.SEARCH_SPACES[model_name]
        pipeline_class = configuration["pipeline"]
        param_distributions = configuration["params"]
        pipeline = pipeline_class().create_pipeline()
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            scoring="neg_root_mean_squared_error",
            cv=self.cv,
            n_jobs=-1,
            random_state=self.random_state,
            return_train_score=True,
        )

        search.fit(X, y)

        return ModelSearch(
            model_name=model_name,
            best_params=search.best_params_,
            best_rmse=-search.best_score_,
            best_estimator=search.best_estimator_,
        )

    def tune_all_models(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> list[ModelSearch]:
        results = []
        for model_name in self.SEARCH_SPACES:
            result = self.tune_model(model_name=model_name, X=X, y=y)
            results.append(result)

        return results

    @staticmethod
    def get_best_model(results: list[ModelSearch]) -> ModelSearch:
        return min(results, key=lambda result: result.best_rmse)


def find_best_model_validation():
    data = load_data()
    X = data.drop(columns=TARGETS)
    y = data[TARGETS]

    x_train, x_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    validator = ModelValidation()
    results = validator.tune_all_models(x_train, y_train)
    best_model = validator.get_best_model(results)

    print(f"Model: {best_model.model_name}")
    print(f"CV RMSE: {best_model.best_rmse:.4f}")
    print(f"Best parameters: {best_model.best_params}")
    y_pred = best_model.best_estimator.predict(x_test)
    print("\nPrediction shape:")
    print(y_pred.shape)


if __name__ == "__main__":
    find_best_model_validation()
