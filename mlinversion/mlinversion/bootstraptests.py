"""
Regularisation Tests

This file contains a set of functions that were used to test and optimise
the effects of regularisation.

This file can also be imported as a module and contains the following
functions:

    * func()
        Blurb about func.

"""
import os
import numpy as np
from copy import deepcopy
from typing import Union
from scipy.sparse import coo_matrix
from lininvbox.lininvbox.inversion import Inversion
from lininvbox.lininvbox.constructors import DesignMatrix, DataArray
from lininvbox.lininvbox.regularisation import Regularisation
from lininvbox.lininvbox.constraints import Constraints


def do_bts_serial(pct: float,
                  _G: DesignMatrix,
                  _d: DataArray,
                  _C: Constraints,
                  _Gamma: Regularisation
                  ) -> np.ndarray:
    """
    TODO
    """

    # set up a unique inversion object
    inv = Inversion("")
    #
    x = np.random.choice(
        _G.matrix.shape[0],
        int(np.ceil(
            np.round(_G.matrix.shape[0] * pct) / 2) * 2), replace=False)

    np.random.shuffle(x)

    keep, replace = x[
        :int(np.round(len(x) / 2, 0))], x[int(np.round(len(x) / 2, 0)):]

    _G = deepcopy(_G)
    _d = deepcopy(_d)

    tG = _G.matrix.toarray()
    td = _d.array.toarray()

    tG[replace] = tG[keep]
    td[replace] = td[keep]

    _G.matrix = coo_matrix(tG)
    _d.matrix = coo_matrix(td)

    m = inv.invert(
        _G, _d, constraints=_C, regularisation=_Gamma, inplace=False)

    return m.array.toarray()[:_G.matrix.shape[1]]


def run_bts(nbts: int,
            pct: float,
            _G: DesignMatrix,
            _d: DataArray,
            _C: Constraints,
            _Gamma: Regularisation
            ) -> np.ndarray:
    """
    TODO.
    """
#     global pct, G, d, constraints, regularisation
    bts_out = np.concatenate(
        [do_bts_serial(pct, _G, _d, _C, _Gamma) for _ in range(nbts)], axis=1)
    return bts_out


def run_bootstrap_session(nbts: int,
                          G: DesignMatrix,
                          d: DataArray,
                          pct: float,
                          how: str = "SERIAL",
                          ow: bool = False,
                          poolsize: bool = None,
                          constraints: Union[None, Constraints] = None,
                          regularisation: Union[None, Regularisation] = None,
                          root: str = "../mlinversion/.bts",
                          ) -> np.ndarray:
    """
    TODO.
    """
    if not ow:
        try:
            return np.load(f"{root}/bts.npy")
        except FileNotFoundError:
            pass

    # TODO: Fix parallel implementation - low priority.
#     if how.upper() == "PARALLEL":
#         with Pool(poolsize) as p:
#             print(f"Running {NBTS} bootstrap iterations in parrallel using {poolsize} processes.")
#             bts = np.concatenate(p.map(run_bts, [10 for _ in range(int(NBTS/10))]), axis=1)
    if how.upper() == "SERIAL":
        print(f"Running {nbts} bootstrap iterations in serial.")
        bts = run_bts(nbts, pct, G, d, constraints, regularisation)

    os.makedirs(f"{root}/", exist_ok=True)
    np.save(f"{root}/bts.npy", bts)

    return bts
