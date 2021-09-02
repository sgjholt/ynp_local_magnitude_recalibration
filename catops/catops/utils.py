"""Utils.py is a module that contains functions that are designed to perfrom
operations that are generally for catalog operations"""

import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
from .plotting import spatial_distribution_plot, magnitude_distance_plot


def concat_dataframes(df1: pd.DataFrame,
                      df2: pd.DataFrame
                      ) -> pd.DataFrame:
    """
    Convenience function to concatenate two dataframes (dfs) together
    throwing out columns that don't exist in both df instances.

    Returns: pd.DataFrame
    """
    return pd.concat([df1, df2], join='inner')


def clear_issues(df: pd.DataFrame,
                 funcs,
                 ) -> None:
    """
    Apply whatever functions you want that perform inplace operations on the
    dataframe.

    Returns: None
    """
    for func in funcs:
        func(df)


def dataframe_difference(df1: pd.DataFrame,
                         df2: pd.DataFrame,
                         which: str = 'both',
                         ) -> pd.DataFrame:
    """Returns the table that contains the set of elements that do 
    not overlap between two dataframes.

    Args:
        df1 (pd.DataFrame): The first DataFrame object.
        df2 (pd.DataFrame): The second DataFrame object.
        which (str, optional): The argument of the set comparison. 
            Defaults to 'both'.

    Returns:
        pd.DataFrame: [description]
    """
    comparison_df = df1.merge(
        df2,
        indicator=True,
        how='outer'
    )
    diff_df = comparison_df[comparison_df['_merge'] != which]

    return diff_df


def print_catalog_stats(df: pd.DataFrame) -> None:
    """Print the earthquake and amplitude count of a given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame of interest.
    """
    EqCount = len(df['Evid'].unique())
    AmpCount = len(df)
    print(f"The catalog contains {AmpCount} amplitudes from {EqCount} earthquakes.")


def quick_inspect_magnitude_distance(df: pd.DataFrame,
                                     save: Union[str, bool] = False,
                                     **kwargs
                                     ) -> None:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        save (Union[str, bool], optional): [description]. Defaults to False.
    """
    inspec = df.copy(deep=True)
    inspec = inspec[inspec.CatMag != -9.99]
    M, Dist, Dep, A = inspec.CatMag, inspec.Rhyp, inspec.drop_duplicates(
        "Evid").EqDep, np.log10(inspec.halfAmpH)

    magnitude_distance_plot(M, Dist, Dep, A)

    if save:
        assert type(save) is str
        plt.savefig(save, **kwargs)
        print(f"saved {save}")


def quick_inspect_spatial_distribution(df: pd.DataFrame,
                                       save: Union[str, bool] = False,
                                       **kwargs,
                                       ) -> None:
    inspec = df.copy(deep=True)
    Lon = inspec.EqLon.values
    Lat = inspec.EqLat.values
    Dep = inspec.EqDep.values
    spatial_distribution_plot(Lon, Lat, Dep)

    if save:
        assert type(save) is str
        plt.savefig(save, **kwargs)
