"""A collection of classes to that build regularisation matrices for a least
squares inversion.

The regularisation matrices are created to be used as part of Tikhonov
regularisation [1]. I think this is also referred to as damping or damped
least squares. In the simplest case the Tikhonov matrix (Γ) is just the
identity (I) matrix multiplied by a smoothing operator (α). Such that Γ = αI.
I can also be replaced with a finite difference or Fourier operator.


This file can also be imported as a module and contains the following
classes:

    * X
        - Y

Notes
-----
    Indentation means class is sub-class of the one in the next highest scope.

References
----------
    [1] https://en.wikipedia.org/wiki/Tikhonov_regularization
"""


import numpy as np
from typing import Union
from collections import OrderedDict
from scipy.sparse import coo_matrix
from .basetypes import Matrix
from .equation import Equation
from .operations import finite_difference_mat


class RegCoeffs(Matrix):
    """
    A class that stores a regularisation matrix spanning the whole Equation.
    """

    def __init__(self,
                 term_map: Equation,
                 matrix: Union[coo_matrix, np.ndarray, None] = None,
                 ):

        super().__init__(term_map=term_map, matrix=matrix)

    def __add__(self, other):

        mat = coo_matrix(self.matrix + other.matrix)
        return RegCoeffs(term_map=self.term_map, matrix=mat)


class SingleRegCoeffs(RegCoeffs):
    """
    A class that allocates and stores a regularisation matrix for a single
    term in the equation.
    """

    func_map = {"IDENTITY": np.identity,
                "FD": finite_difference_mat,
                }

    def __init__(self,
                 term_map: Equation,
                 kind: str,
                 name: str,
                 ):
        super().__init__(term_map=term_map)

        self.allocate_matrix(self.__reg_alloc,
                             term_map,
                             kind=kind,
                             name=name,
                             shape=(term_map.npars, term_map.npars),
                             )

    def __get_func(self, kind: str):
        if kind.upper() not in self.func_map.keys():
            msg = f"{kind} option not available," +\
                f"choose from {self.func_map.keys()}"
            raise KeyError(msg)
        return self.func_map[kind.upper()]

    def __reg_alloc(self,
                    term_map: Equation,
                    kind: str,
                    name: str,
                    ) -> Union[np.ndarray,
                               np.ndarray,
                               np.ndarray
                               ]:

        func = self.__get_func(kind)
        tm = term_map.values[name]
        size = len(tm['model_indices'])
        shift = tm['model_indices'][0]

        mat = func(size)
        bmat = coo_matrix(mat)

        # this clause is required because later in the pipeline the
        # reg matrix is multiplied by its transpose (gamma.T @ gamma) and alpha
        # will be multiplied.

        return ((bmat.row + shift).astype(int), (bmat.col + shift).astype(int),
                bmat.data)


class Regularisation():
    """
    A class that handles constraints, then builds and stores
    constraint coefficient (F) and data (h) arrays
    """

    def __init__(self,
                 term_map: Equation,
                 regs: Union[dict, OrderedDict]
                 ):

        self.term_map = term_map

        if regs:  # make sure dict is actually populated
            self.regs = regs
        else:
            print("WARNING empty constraints, nothing done.")

        self.__assign_constraint_matrices()

    def __assign_constraint_matrices(self):
        for param, regs in self.regs.items():
            if param in self.term_map.values.keys():
                sreg = SingleRegCoeffs(term_map=self.term_map,
                                       name=param,
                                       kind=regs['kind'],
                                       )
                try:
                    self.gamma = self.gamma + sreg
                    self.alpha.update({param: self.__get_alpha(regs)})
                except AttributeError:  # If there isn't one already.
                    self.gamma = sreg
                    self.alpha = {param: self.__get_alpha(regs)}
                self.term_map.values[param]['regularisation'].update(dict(
                    **regs))
            else:
                print(f"WARNING: {param} not in Equation.")
                print("Skipping ...")
                continue

    def __get_alpha(self, regs: dict) -> Union[int, float]:
        if {'alpha', } <= regs.keys():
            alpha = regs['alpha']
        else:
            alpha = 1
        return alpha

    @property
    def regs(self) -> OrderedDict:
        return self._regs

    @regs.setter
    def regs(self, rgs: Union[dict, OrderedDict]):
        rgs = OrderedDict(rgs)
        self._regs = rgs

    @property
    def term_map(self) -> Equation:
        return self._term_map

    @term_map.setter
    def term_map(self, tm: Equation):
        assert type(tm) is Equation
        self._term_map = tm

    @property
    def gamma(self) -> Union[RegCoeffs, SingleRegCoeffs]:
        return self._gamma

    @gamma.setter
    def gamma(self, gam):
        assert issubclass(type(gam), RegCoeffs), "Regularisation" + \
            f"coefficients must be of type {type(RegCoeffs)}"
        self._gamma = gam

    @property
    def alpha(self) -> dict:
        return self._alpha

    @alpha.setter
    def alpha(self, a: dict):
        assert type(a) is dict, "alpha must be like dict(name=alpha)"
        assert set(a.keys()) <= self.term_map.values.keys(), \
            f"{set(a.keys())} do not all exist in Equation."
        self._alpha = a
