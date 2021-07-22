"""A collection of functions/classes to parse UUSS station corrections file.

This file can also be imported as a module and contains the following
classes:

    * X
        - Y

Note: Indentation means it is a sub-class of the object in the next highest
scope.
"""
import os
import numpy as np
import pandas as pd
from typing import Union, Callable

# PATH NAME TO HERE
PDIR = os.path.dirname(os.path.abspath(__file__))
UUSTACORS = "data/sta_cors_P07.txt"
PATH = os.path.join(PDIR, UUSTACORS)
COL_MAP = {"sta": "Sta-UUSS", "corr": "Sj-UUSS"}


class UUSJ():

    def __init__(self, path: str = PATH):
        self.sj = self.load_sj(path)

    def load_sj(self, path: str):
        uuss_sj = pd.read_csv(path, delim_whitespace=True)\
            .rename(columns=COL_MAP)
        uuss_sj.loc[uuss_sj['config'].isin(['exclude', '0']), 'Sj-UUSS'] = np.nan
        uuss_sj = uuss_sj[uuss_sj.seedchan.str.endswith("E")]
        uuss_sj.drop_duplicates("Sta-UUSS", inplace=True)
        return uuss_sj[list(COL_MAP.values())].reset_index(drop=True)
