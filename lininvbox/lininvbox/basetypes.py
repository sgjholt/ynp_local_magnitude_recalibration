"""Matrix Base Classes

This file contains the base Matrix classes that represent a linear
equation Gm=d. Where G is the design matrix of coefficients, m are the
model parameters and d is the data array of observations.

This file contains the following classes:

    * DesignMatrix - handles coefficients of an arbitrary linear equation
    * ModelMatrix - handles the model parameters
    * DataMatrix - handles the data array
"""
import numpy as np
from typing import Union, Tuple
from .equation import Equation, Term
from scipy.sparse import vstack, coo_matrix


class Matrix():
    """
    A class used as a base for all matrix like objects.

    Sub-classes
    -----------
    DesignMatrix
        The coefficients matrix (G) of a matrix equation like Gm=d.

    Array
        A matrix-like container, but expects a 1 dimensional input.
    ...

    Attributes
    ----------
    term_map :  Union[.equation.Equation, .equation.Term]
        An object that handles the mapping of parameter labels to recovered
        values.
    matrix : scipy.sparse.coo.coo_matrix or np.ndarray
        a sparse matrix of coefficients for an arbitrary linear equation

    Methods
    -------
    N/A
    """

    def __init__(self,
                 term_map: Union[Equation, Term, None] = None,
                 matrix: Union[coo_matrix, np.ndarray, None] = None,
                 ):
        if term_map is not None:
            self.term_map = term_map
        # Got to create placeholder coo_matrix for each new instance of Matrix
        # ... if one is not passed when Matrix is initiated.
        if matrix is None:
            matrix = coo_matrix(([0, ], ([2, ], [2, ])))

        self.matrix = matrix

    def allocate_matrix(self,
                        alloc_func,
                        term_map: Union[Equation, Term],
                        inplace: bool = True,
                        shape: Union[Tuple[int, int], None] = None,
                        **kwargs,
                        ) -> coo_matrix:
        """
        This function applies an arbitrary mapping function to values in the
        term map to return three numpy arrays as a tuple that describe a
        triplet matrix COOrdinate system.

        ...

        Parameters
        ----------


        """
        try:
            rows, cols, vals = alloc_func(term_map, **kwargs)
        except ValueError:
            msg = "Allocation function must return three np.ndarrays as a" +\
                "tuple of values, row numbers and column numbers"
            raise ValueError(msg)

        tmat = coo_matrix((vals, (rows, cols)), shape=shape)

        if not inplace:
            return tmat

        self.matrix = tmat

    # Getters and Setters using @property decorator.
    @property
    def matrix(self) -> coo_matrix:
        """
        Matrix attribute getter.

        Returns
        -------
        """
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: coo_matrix):
        """
        Matrix attribute setter. Checks to see if the set object is a
        coo_matrix.

        ...

        Parameters
        ----------
        term_map :  .mappers.ParamMap
            An object that handles the mapping of parameter labels to recovered
            values.


        Raises
        ------
            Raises assertion error if a matrix is is not of type
            scipy.sparse.coo.coo_matrix.
        """
        assert type(matrix) is coo_matrix or type(matrix) is np.ndarray,\
            f"Matrix must be of type {coo_matrix} or {np.ndarray} not {type(matrix)}."
        if type(matrix) is np.ndarray:
            matrix = coo_matrix(matrix)
        self._matrix = matrix

    @property
    def term_map(self) -> Union[Equation, Term]:
        """
        Parameter map attribute getter.
        """
        return self._term_map

    @term_map.setter
    def term_map(self, term_map: Union[Equation, Term]):
        """
        Matrix attribute setter. Checks to see if the set object is a
        coo_matrix.

        ...

        Parameters
        ----------
        term_map :  .mappers.ParamMap
            An object that handles the mapping of parameter labels to recovered
            values.


        Raises
        ------
            Raises assertion error if a ParamMap object is not passed.
        """
        assert issubclass(type(term_map), Equation),\
            f"Parameter map must be of type {Equation} or type Term."
        self._term_map = term_map


class Array(Matrix):
    """

    """

    def __init__(self,
                 term_map: Union[Equation, Term, None] = None,
                 array: Union[coo_matrix, None] = None,
                 ):

        super().__init__(term_map, array)

        if array is None:
            array = coo_matrix(([0, ], ([2, ], [0, ])))

        else:
            self.array = array

    @property
    def array(self):  # array is just an alias for matrix
        return self._matrix

    @array.setter
    def array(self, arr: coo_matrix):  # perform one additional check on shape
        assert arr.shape[0] == 1 or arr.shape[1] == 1,\
            f"array must be 1 dimensional not {arr.shape}."
        if arr.shape[1] != 1:  # ensure that the shape is always like this.
            arr = arr.reshape(arr.shape[::-1])
        self.matrix = arr

    def append(self, arr, inplace: bool = False):
        """
        Appends an Array to the current Array or returns a new one.
        """
        newarray = vstack((self.array, arr.array))
        if not inplace:
            try:
                return Array(self.term_map, newarray)
            except AttributeError:
                return Array(array=newarray)
        else:
            self.array = newarray
