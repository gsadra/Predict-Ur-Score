import pandas as pd

from predict_ur_score.utils import load_data


class EDA:
    """A class to perform exploratory data analysis (EDA) on the student scores data."""

    def __init__(self) -> None:
        self.df: pd.DataFrame = load_data()

    def data_shape(self) -> tuple:
        return self.df.shape

    def data_columns(self) -> list:
        return self.df.columns.tolist()

    def data_dtypes(self) -> pd.Series:
        return self.df.dtypes

    def nan_counts(self) -> pd.Series:
        return self.df.isna().sum()

    def null_counts(self) -> pd.Series:
        return self.df.isnull().sum()

    def career_aspirations(self) -> list:
        return self.df["career_aspiration"].unique().tolist()


if __name__ == "__main__":
    eda = EDA()
    print("Career Aspirations:", eda.career_aspirations())

    df = load_data()
    print(
        "Categorical Columns:",
        df.select_dtypes(include=["str"]).columns.tolist(),
    )
