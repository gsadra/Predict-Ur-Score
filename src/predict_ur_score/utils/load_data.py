import pandas as pd
from pathlib import Path
from predict_ur_score.exceptions import PredictorFileNotFound, PredictorInvalidData

def load_data(path: str | Path ='data/student-scores.csv', head: bool = False) -> pd.DataFrame:
    '''
    Load data from a CSV file into a pandas DataFrame.

    Args:
        path (str): The file path to the CSV file.
        head (bool): If True, print the first five rows of the DataFrame.

    Returns:
        pandas.DataFrame: The loaded data as a pandas DataFrame.
    '''

    try:
        df = _select_data(pd.read_csv(path))
        if head:
            print(df.head())

        return df

    except FileNotFoundError:
        raise PredictorFileNotFound(f'File not found at {path}')
    
    except Exception as e:
        raise PredictorInvalidData(f'Error loading data from {path}: {e}')

def _select_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = dataframe
    df.drop(['id', 'first_name', 'last_name', 'email'], axis=1, inplace=True)

    return df