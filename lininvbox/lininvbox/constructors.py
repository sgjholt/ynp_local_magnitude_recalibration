"""Design Matrix Constructor Functions

This file contains the classes

This file can also be imported as a module and contains the following
classes:

    * LinInterpCoeffs
    * ConstantCoeffs
    * ModelArray
    * DataArray


"""
import numpy as np
from typing import Tuple, Union
from scipy.sparse import hstack, coo_matrix

from .equation import Equation, Term
from .basetypes import Matrix, Array
from .operations import (build_interp_coeffs_as_triplet,
                         build_constant_coeffs_as_triplet
                         )


class DesignMatrix(Matrix):
    """
    A container for design matrix coefficients.
    It tracks and stores model parameter maps that are tied to the
    type of the design matrix sub-classes. It is useful to track
    these values because the design matrix coefficients are related
    to a linear equation with an arbitrary number of model parameters.

    ...

    Sub-classes
    -----------
    LinInterpCoeffs
        Blah.

    ConstCoeffs
        Blah.

    Attributes
    ----------
    see .basetypes.Matrix

    Methods
    -------
    N/A
    """

    def __init__(self,
                 term_map: Union[Equation, Term],
                 matrix: Union[coo_matrix, None] = None,
                 ):
        super().__init__(term_map, matrix)

    def __add__(self, other):

        assert issubclass(type(other), Matrix), "OBJECTION!"
        new_pmap = self.term_map + other.term_map
        new_mat = hstack((self.matrix, other.matrix))
        return DesignMatrix(new_pmap, new_mat)


class LinInterpCoeffs(DesignMatrix):
    """
    A subclass of DesignMatrix. This class is specific to the allocation
    and storage of interpolation coefficients. The allocation of interpolation
    coefficients is slightly more complex because the coefficients are computed
    in adjacent pairs for each row. The rows are determined based on the
    relationship of the observed label to the interpolation labels (or nodes).
    The coefficients size depends on the linear interpolation function that
    is passed.
    ...
    # Note, could you theoretically replace the linear interpolation func
    # with a non-linear version to do a non-linear interpolation?
    ...

    Attributes
    ----------

    See '.basetypes.DesignMatrix'.

    Methods
    -------

    N/A
    """

    def __init__(self, term_map: Term):
        super().__init__(term_map)
        self.allocate_matrix(self.__interp_mat_alloc, term_map=term_map)

    def __interp_mat_alloc(self, term_map: Term) -> coo_matrix:

        # term map should have only one key
        key = list(term_map.values.keys())[0]
        labels = term_map.values[key]['raw_labels']
        nodes = term_map.values[key]['unique_labels']
        c = term_map.values[key]['sign']
        # compute the matrix coordinates (COO)
        rows, cols, vals = build_interp_coeffs_as_triplet(labels, nodes, c)
        # convert to scipy.sparse.coo.coo_matrix object and
        # print(vals, rows, cols)
        # print(list(x.dtype for x in (vals, rows, cols)))
        return rows, cols, vals


class ConstantCoeffs(DesignMatrix):
    """
    A subclass of DesignMatrix. This class is specific to the allocation
    and storage of coefficients that are a constant value (normally = 1).
    ...

    Attributes
    ----------

    See '.basetypes.DesignMatrix'.

    Methods
    -------

    N/A
    """

    def __init__(self, term_map: Term):
        super().__init__(term_map)
        self.allocate_matrix(self.__const_mat_alloc, term_map=term_map)

    def __const_mat_alloc(self, term_map: Term) -> Tuple[np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray
                                                         ]:

        # term map should have only one key at this entry point
        key = list(term_map.values.keys())[0]
        rlabs = term_map.values[key]['raw_labels']
        ulabs = term_map.values[key]['unique_labels']
        uinds = term_map.values[key]['unique_indices']
        c = term_map.values[key]['sign']
        # compute the matrix coordinates (COO)
        rows, cols, vals = build_constant_coeffs_as_triplet(
            uinds, ulabs, rlabs, c)
        # convert to scipy.sparse.coo.coo_matrix object and
        # print(vals, rows, cols)
        # print(list(x.dtype for x in (vals, rows, cols)))
        return rows, cols, vals


class DataArray(Array):
    """
    The DataArray class is a sub-class of array that is basically just a
    fancy container for the observations. DataArray knows how to construct
    an Array object in the required format.
    ...

    Attributes
    ----------
    array : coo_matrix
        An array (as a coo_matrix) of the data.

    Methods
    -------
    WRITE ME.

    """

    def __init__(self, data: np.ndarray):
        super().__init__()
        self.array = coo_matrix(data)


class ModelArray(Array):
    """
    WRITE ME.
    ...

    Attributes
    ----------
    WRITE ME.

    Methods
    -------
    WRITE ME.

    """

    def __init__(self, term_map: Equation, model: np.ndarray):
        super().__init__(term_map)
        self.array = coo_matrix(model)
        self.__dump_model_values()

    def __dump_model_values(self) -> None:
        tm = self.term_map.values
        for group in tm.keys():
            tm[group]['model_values'] = self.get_group_m(group)

    def get_group_m(self, name: str):
        array = self.array.toarray().flatten()
        return array[self.term_map.values[name]['model_indices']]
