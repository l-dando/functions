import pandas as pd
import numpy as np
import copy
from scipy import spatial


def infill(
        df_in: pd.DataFrame,
        axis: list = ['x', 'y'],
        cols: list = [],
) -> pd.DataFrame:
    """
        Uses a Nearest Neighbours KD-Tree to infill missing data based off of latitude and longitude
        :param df_in: pandas dataframe containing the data (both nulls and non-nulls)
        :param axis: the names of columns to be used to construct your tree
        :param cols: the columns to infill
        :return: The original DataFrame in original order with nulls infilled
        """

    df = copy.deepcopy(df_in)

    column_order = df.columns.values

    x, y = axis

    df['ind'] = 1
    df['ind'] = df['ind'].cumsum() - 1

    df_nonulls = df.dropna(axis=0, subset=cols).reset_index(drop=True)
    df_nulls = df.iloc[list(set(df['ind']).difference(set(df_nonulls['ind']))), :].reset_index(drop=True)

    kd_tree = spatial.KDTree(np.c_[df_nulls[x].ravel(), df_nulls[y].ravel()])

    dd, ii = kd_tree.query([df_nulls[[x, y]].to_numpy()], k=1, workers=-1, p=2)

    df_nulls['ref'] = pd.Series(ii[0])

    df_nulls.loc[:, cols] = df_nulls.loc[:, cols].combine_first(
        df_nonulls.iloc[df_nulls['ref'].iloc[:]][cols].reset_index(drop=True))

    df = pd.concat([df_nonulls, df_nulls])
    df = df.sort_values('ind', ascending=True).reset_index(drop=True)
    df = df[column_order]

    return df


def smoothing(
        df_in: pd.DataFrame,
        axis: list = ['x', 'y'],
        col: list = [],
        n: int = 500,
        p: int = 0.5,
) -> pd.DataFrame:
    """
        Uses a Nearest Neighbour KD-Tree to smooth values by using the distances and values of the n-closest samples in a 2D space
        :param df_in: pandas dataframe containing the data
        :param axis: the names of columns to be used to construct your tree
        :param col: the column you wish to smooth
        :param n: the number of nearest neighbours to use in smoothing
        :param p: the strength of the smoothing
        :return: original dataframe alongside new smoothed column
        """
    df_ref = copy.deepcopy(df_in)
    df_smooth = copy.deepcopy(df_in)

    x, y = axis

    kd_tree = spatial.KDTree(np.c_[df_ref[x].ravel(), df_ref[y].ravel()])

    dd, ii = kd_tree.query([df_smooth[[x, y]].to_numpy()], k=n + 1, workers=-1, p=2)

    df_ref['ref_ind'] = ii[0][:, 1:].tolist()
    df_ref['ref_dist'] = dd[0][:, 1:].tolist()

    df_ref['ref_vals'] = df_ref['ref_ind'].apply(lambda x: df_ref.iloc[x][col].values)

    df_ref['ref_weights'] = (1 - (1 / df_ref['ref_dist'].apply(sum) * df_ref['ref_dist'].apply(np.array))) / (n - 1)

    df_ref['neighbour_vals'] = [np.dot(x, y) for x, y in zip(df_ref.ref_vals, df_ref.ref_weights)]

    df_smooth[col + '_smooth'] = (p * df_ref[col]) + ((1 - p) * df_ref['neighbour_vals'])

    return df_smooth
