"""
This module contains a set of constant values that are relevent to geographic
balancing of events for a 'clean' amplitude database.
"""
from .constants import GeoBalanceConstants as C
import numpy as np
import pandas as pd
from typing import Union, Tuple


def slices(min: float, max: float, N: float) -> np.array:
    """
    Convenience function that calls np.linspace and returns
    a numpy array of N linearly spaced values between min, max.

    Returns: np.array
    """
    assert N > 1, "Need at least two points between start and end. Nmin is 2."

    return np.linspace(min, max, N)


def reduce_event_count_per_voxel(df: pd.DataFrame, EV_MAX: int = C.EV_MAX):
    """
    Convenience function to sort and reduce the event count in any
    given voxel
    """
    return df.sort_values(["voxel", "Namps"], ascending=False)\
                           .groupby('voxel')\
                           .head(EV_MAX)


def grab_voxel_info(
    voxels: dict,
    X: np.array,
    Y: np.array,
    Z: np.array,
    i: int,
    j: int,
    k: int,
    count: int) -> None:
    """
    Grabs the indexes and creates detailed voxel info which is
    appended to a dictionary for safe keeping.

    Returns None
    """

    voxels['Xrange'].append((X[i], X[i+1]))
    voxels['Yrange'].append((Y[j], Y[j+1]))
    voxels['Zrange'].append((Z[k], Z[k+1]))
    voxels['Xmid'].append(np.median((X[i], X[i+1])))
    voxels['Ymid'].append(np.median((Y[j], Y[j+1])))
    voxels['Zmid'].append(np.median((Z[k], Z[k+1])))
    voxels['Vnum'].append(count)


class Voxels():
    """
    Voxels object stores all information related to
    voxels for slices in X, Y, Z that are given
    when an instance of Voxels is created. The default
    values are given in the constants file, but can
    be overridden.

    Attributes:
        slices: dict: {'X': np.array, 'Y':np.array, 'Z':np.array}
        voxels: pd.DataFrame containing voxel spatial information and labels

    Methods:

    Returns: Voxel
    """

    def __init__(self,
                 x_slices: np.array = slices(C.LON_MIN, C.LON_MAX, C.LON_SLICES),
                 y_slices: np.array = slices(C.LAT_MIN, C.LAT_MAX, C.LAT_SLICES),
                 z_slices: np.array = slices(C.DEP_MIN, C.DEP_MAX, C.DEP_SLICES)
                 ):

        self.set_slices(x_slices, y_slices, z_slices)
        self.voxels = pd.DataFrame(dict(Xrange=[], Yrange=[], Zrange=[],
                                            Xmid=[], Ymid=[], Zmid=[], Vnum=[]))


    def set_slices(self, x:np.array, y:np.array, z:np.array) -> None:
        """

        """

        self.slices=dict(X=x, Y=y, Z=z)

    # this method is kind of messy but could be refactored at another point
    def assign_voxels(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Function that iterates over the volumetric pixels (voxels)
        for the total volume that is segmeneted into smaller volumes
        according to the slices passed in X, Y and Z directions.
        It assigns each voxel a number from 1 to N=XYZ voxels.

        Returns: None
        """

        df = df.copy(deep=True)

        X, Y, Z = self.slices['X'], self.slices['Y'], self.slices['Z']

        cell = []
        bounds = []
        count = 0
        voxel_locs = dict(Xrange=[], Yrange=[], Zrange=[],
                          Xmid=[], Ymid=[], Zmid=[], Vnum=[]
                          )

        # brute force search for events in each sub-volume over entire domain
        # a little slow but can think of ways to optimise later since it works!
        for i in range(len(X[:-1])):
            for j in range(len(Y[:-1])):
                for k in range(len(Z[:-1])):

                    grab_voxel_info(voxel_locs, X, Y, Z, i, j, k, count)
                    cell.append(count)
                    bounds.append(
                                 (df['EqLon']>X[i])&(df['EqLon']<=X[i+1])&\
                                 (df['EqLat']>Y[j])&(df['EqLat']<=Y[j+1])&\
                                 (df['EqDep']>Z[k])&(df['EqDep']<=Z[k+1])
                                  )
                    count += 1

        self.voxels = pd.DataFrame(voxel_locs)

        df['voxel'] = np.select(bounds, cell)

        return df
