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
from typing import Union, Tuple
from collections import OrderedDict
from lininvbox.lininvbox.inversion import Inversion
from lininvbox.lininvbox.constructors import DesignMatrix, DataArray
from lininvbox.lininvbox.regularisation import Regularisation
from lininvbox.lininvbox.constraints import Constraints
from lininvbox.lininvbox.operations import roughness, mse
from lininvbox.lininvbox.utils import delete_directory

# Constant thread limitation for inversions.
THREAD_LIMIT = 2
THREAD_LIMIT_API = 'blas'


# functions for project
def reg_test_invert(inv: Inversion,
                    G: DesignMatrix,
                    d: DataArray,
                    alpha: float,
                    constraints: Union[None, Constraints] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """

    """
    Gamma = Regularisation(G.term_map,
                           regs=OrderedDict(logA0n=dict(kind="FD",
                                                        alpha=alpha)))

    m = inv.invert(G,
                   d,
                   regularisation=Gamma,
                   constraints=constraints,
                   inplace=False)

    fd = inv.forward(G, m)

    _mse = mse(d.array.A.flatten(), fd.array.A.flatten())
    inds = m.term_map.values["logA0n"]['model_indices']
    _rough = roughness(m.array.A[inds].flatten())

    return _mse, _rough


def regularisation_optimiser(inv: Inversion,
                             G: DesignMatrix,
                             d: DataArray,
                             alphas: np.ndarray,
                             constraints: Union[None, Constraints] = None
                             ) -> Tuple[np.ndarray, np.ndarray, float]:
    """

    """
    _rough = np.zeros(len(alphas))
    _mse = np.zeros(len(alphas))

    for i in range(len(alphas)):
        MSE, ROUGH = reg_test_invert(inv, G, d, alphas[i], constraints)
        _mse[i] = MSE
        _rough[i] = ROUGH

    # obtain turning point of "L-curve" (minimum in this case) ...
    # and take that as the optimal value for alpha.
    # + 2 because you lose two points by differentiating twice from the ...
    # finite difference approximations
    pt = np.abs(np.diff(np.log10(_rough), 2)).argmin() + 2

    return _mse, _rough, pt


def do_norm_test(inv: Inversion,
                 G: DesignMatrix,
                 d: DataArray,
                 alphas: np.ndarray,
                 root: str = "../mlinversion/.norm"
                 ) -> Tuple[np.ndarray, np.ndarray, float]:
    """

    """
    try:
        a_comp = np.load(f"{root}/alphas.npy")
        if not np.array_equal(a_comp, alphas):
            print("new alphas detected...")
            delete_directory(f"{root}/")
        print("alphas unchanged...")
        _mse = np.load(f"{root}/mse.npy")
        _rough = np.load(f"{root}/rough.npy")
        best_i = np.load(f"{root}/besti.npy")
        print("loaded local files...")

    except FileNotFoundError:
        print("running inversions...")
        _mse, _rough, best_i = regularisation_optimiser(inv, G, d, alphas)
        os.makedirs(f"{root}/", exist_ok=True)
        np.save(f"{root}/mse.npy", _mse)
        np.save(f"{root}/rough.npy", _rough)
        np.save(f"{root}/besti.npy", best_i)
        np.save(f"{root}/alphas.npy", alphas)
    print(f"Best alpha is {alphas[best_i]:.2f}.")

    return _mse, _rough, best_i
