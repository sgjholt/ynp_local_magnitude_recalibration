"""Design Matrix Constructor Functions

This file contains the classes that pertain to building constraints for use
that are passed to the Inversion class.

This file can also be imported as a module and contains the following
classes:

    * ConstraintCoeffs
        - SingleConstraintCoeffs
"""
import numpy as np
from typing import Union
from scipy.sparse import vstack, coo_matrix
from collections import OrderedDict
from .basetypes import Matrix, Array
from .constructors import DataArray
from .equation import Equation
from .operations import const_constraint_coeffs, sum_constraint_coeffs


class ConstraintsCoeffs(Matrix):
    """
    A class to store constraints.
    """

    def __init__(self,
                 term_map: Equation,
                 matrix: Union[np.ndarray, None] = None
                 ):
        super().__init__(term_map, matrix)

    def stack(self, con, inplace: bool = False):
        """
        Appends a constraints matrix to the another one and returns a new
        object.
        """
        newmatrix = vstack((self.matrix, con.matrix))
        if not inplace:
            return ConstraintsCoeffs(self.term_map, matrix=newmatrix)
        else:
            self.matrix = newmatrix


class SingleConstraintsCoeffs(ConstraintsCoeffs):
    """
    A container for constraint coefficients. It will allocate a constraint
    coefficient matrix from the data label that is passed. Constraints may be
    chanined together.

    ...

    Sub-classes
    -----------

    Attributes
    ----------
    see .basetypes.Matrix

    Methods
    -------
    N/A
    """

    func_map = {"CONSTANT": const_constraint_coeffs,
                "SUM": sum_constraint_coeffs,
                }

    def __init__(self,
                 term_map: Equation,
                 name: str,
                 kind: str,
                 label: Union[np.ndarray, None] = None,
                 ):

        super().__init__(term_map)

        self.allocate_matrix(self.__constraint_alloc,
                             term_map,
                             kind=kind,
                             name=name,
                             label=label,
                             shape=(1, term_map.npars),
                             )

    def __get_func(self, kind: str):
        if kind.upper() not in self.func_map.keys():
            msg = f"{kind} option not available," +\
                f"choose from {self.func_map.keys()}"
            raise KeyError(msg)
        return self.func_map[kind.upper()]

    def __constraint_alloc(self,
                           term_map: Equation,
                           kind: str,
                           name: str,
                           label: Union[np.ndarray, None]
                           ) -> Union[np.ndarray,
                                      np.ndarray,
                                      np.ndarray
                                      ]:

        if kind == "CONSTANT" and label is None:
            raise ValueError("Label cannot be 'None' for a single constraint.")

        func = self.__get_func(kind)
        tm = term_map.values[name]
        model_indices = tm['model_indices']
        unique_labels = tm['unique_labels']
        return func(unique_labels=unique_labels,
                    model_indices=model_indices,
                    label=label)


class Constraints():
    """
    A class that handles constraints, then builds and stores
    constraint coefficient (F) and data (h) arrays
    """

    __KEYKINDS = ("SUM", )

    def __init__(self,
                 term_map: Equation,
                 constraints: Union[dict, OrderedDict]
                 ):

        self.term_map = term_map

        if constraints:  # make sure dict is actually populated
            self.constraints = constraints
        else:
            print("WARNING empty constraints, nothing done.")

        self.__assign_constraint_matrices()

    def __assign_constraint_matrices(self):
        for param, cons in self.constraints.items():
            if param in self.term_map.values.keys():
                for conlab, conval in cons.items():
                    if conlab in self.__KEYKINDS:
                        kind = conlab
                    else:
                        kind = "CONSTANT"

                    scons = SingleConstraintsCoeffs(term_map=self.term_map,
                                                    name=param,
                                                    kind=kind,
                                                    label=conlab,
                                                    )
                    svals = DataArray(np.array([conval]))

                    try:
                        self.F = self.F.stack(scons)
                        self.h = self.h.append(svals)
                    except AttributeError:  # If there isn't one already.
                        self.F = scons
                        self.h = svals
                    self.term_map.values[param]['constraints'].update(
                        {conlab: conval})
            else:
                print(f"WARNING: {param} not in Equation.")
                print("Skipping ...")
                continue

    def __repr__(self):
        out = """Fixed... """
        for key, val in self.constraints.items():
            for k in val.keys():
                if str(k).upper() == "SUM":
                    out += key + " " + k + " "
                else:
                    continue
            out += key
        return out

    @property
    def constraints(self) -> OrderedDict:
        return self._constraints

    @constraints.setter
    def constraints(self, cons: Union[dict, OrderedDict]):
        tdict = OrderedDict(cons)
        if tdict:
            for key, value in tdict.items():
                assert issubclass(type(value), dict)
        self._constraints = cons

    @property
    def term_map(self) -> Equation:
        return self._term_map

    @term_map.setter
    def term_map(self, tm: Equation):
        assert type(tm) is Equation
        self._term_map = tm

    @property
    def F(self) -> Union[ConstraintsCoeffs, SingleConstraintsCoeffs]:
        return self._F

    @F.setter
    def F(self, f):
        assert issubclass(type(f), ConstraintsCoeffs), "Constraints" + \
            f"coefficients must be of type {type(ConstraintsCoeffs)}"
        self._F = f

    @property
    def h(self) -> Union[Array, DataArray]:
        return self._h

    @h.setter
    def h(self, H):
        assert issubclass(type(H), Array), "Constraints data" + \
            f"must be of type {type(Array)}"
        self._h = H
