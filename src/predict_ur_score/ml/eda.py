import numpy as np
import pandas as pd
from predict_ur_score.utils import load_data


def eda() -> None:
    df: pd.DataFrame = load_data()

    print('Data Shape: ', df.shape, '\n', )
    print('Data Columns:\n', np.asanyarray(df.columns), '\n', sep='')
    print('Data Data Types:\n', df.dtypes, '\n', sep='')
    print('NaN Counts: \n', df.isna().sum(), '\n', sep='')
    print('Null Counts: \n', df.isnull().sum(), '\n', sep='')

if __name__ == '__main__':
    eda()