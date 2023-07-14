import pandas as pd


def data_smoothing(
        df_in: pd.DataFrame
        axis: list=['latitude', 'longitude'],
        col: list=[],
        n: int=500,
        p: int=0.5,
) -> pd.DataFrame

