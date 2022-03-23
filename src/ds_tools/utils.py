import math

from pandas import DataFrame
from sklearn.model_selection import TimeSeriesSplit


def get_variable_types(df: DataFrame) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for column in df.columns:
        uniques = df[column].dropna(inplace=False).unique()
        if len(uniques) == 2:
            variable_types['Binary'].append(column)
            df[column].astype('bool')
        elif df[column].dtype == 'datetime64':
            variable_types['Date'].append(column)
        elif df[column].dtype == 'int':
            variable_types['Numeric'].append(column)
        elif df[column].dtype == 'float':
            variable_types['Numeric'].append(column)
        else:
            df[column].astype('category')
            variable_types['Symbolic'].append(column)
    return variable_types


def split_timeseries(df, train_size):
    split_point = math.floor(len(df) * train_size)
    train = df[:split_point]
    test = df[split_point:]
    return train, test
