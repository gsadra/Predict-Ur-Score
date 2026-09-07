import pandas as pd
from tabulate import tabulate

from predict_ur_score.exceptions import *
from predict_ur_score.utils import load_data


class Stats:
    """A class to calculate statistics for student scores."""

    TOPICS: list = [
        "math_score",
        "history_score",
        "physics_score",
        "chemistry_score",
        "biology_score",
        "english_score",
        "geography_score",
    ]

    def __init__(self, dataframe: pd.DataFrame = None) -> None:
        self._dataframe = load_data() if dataframe is None else dataframe

    def all_stats(self) -> pd.DataFrame:
        try:
            return self._dataframe.describe().drop(["count"])
        except Exception as e:
            raise PredictorInvalidData(
                f"Error occurred while calculating all statistics: {e}"
            )

    def each_stat(self, column_name: str) -> pd.Series:
        try:
            return self._dataframe[column_name].describe().drop(["count"])
        except Exception:
            raise PredictorInvalidTopic(f"Invalid topic: {column_name}.")

    def table_format(
        self, column_name: str | None = None, all_stats: bool = True
    ) -> str:
        if all_stats:
            series: pd.DataFrame = self.all_stats()
        else:
            series: pd.DataFrame = self.each_stat(column_name).to_frame()

        return tabulate(
            series,
            headers="keys",
            tablefmt="psql",
            numalign="center",
            floatfmt=".2f",
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @dataframe.setter
    def dataframe(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            raise PredictorInvalidInput(
                "Input must be a pandas DataFrame."
            )
        self._dataframe = value
