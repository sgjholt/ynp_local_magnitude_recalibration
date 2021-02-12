"""Inversion Tools

This file contains the classes that compute least squares inversions using data
stored in DesignMatrix and DataArray objects.

This file can also be imported as a module and contains the following
classes:

    * Inversion

"""
import numpy as np
from typing import Union
from scipy.linalg import lstsq
from scipy.sparse import coo_matrix, vstack, hstack
from .operations import compress_matrices, apply_constraints
from .constructors import DesignMatrix, DataArray, ModelArray
from .constraints import Constraints
from .regularisation import Regularisation
from .equation import Equation
from .utils import get_timestamp_now

# TODO: Write constraints containter, add_constraints and __collect_constraints


class Inversion():
    """
    A class used to perform least squares inversions from DesignMatrix
    and DataArray objects.

    ...

    Attributes
    ----------
    G :  .constructors.DesignMatrix
        A sparse matrix of coefficients for an arbitrary linear equation
        of form Gm=d.

    d : .constructors.DataArray
        An array of data for an arbitrary linear equation.

    m : .constructors.ModelArray
        An array of recovered model parameters as returned from
        scipy.linalg.lstqr.

    Methods
    -------
    N/A
    """

    def __init__(self, name: str):
        self.name = name
        self.id = "-".join((name, get_timestamp_now()))

    def invert(self,
               G: DesignMatrix,
               d: DataArray,
               inplace: bool = True,
               constraints: Union[Constraints, None] = None,
               regularisation: Union[Regularisation, None] = None,
               ) -> Union[None, ModelArray]:

        """
        This function takes the DesignMatrix (G) and DataArray (d) and passes
        them to a least-squares inversion (scipy.linalg.lstsq) function to
        recover the ModelArray (m) of a Gm=d type matrix equation. It can
        return m if inplace=False, which can be useful in certain cases
        (e.g. when bootstrapping). The constraints must be passed as a
        Constraints (see .constrains.Constraints) object.

        Parameters
        ----------
        G :  .constructors.DesignMatrix
            A sparse matrix of coefficients for an arbitrary linear equation
            of form Gm=d.

        d : .constructors.DataArray
            An array of data for an arbitrary linear equation.

        inplace : bool = True
            Specifies wether or not to return the output (m) an array of model
            parameters as an .constructors.ModelArray or to assign G, m and d
            back to the current instance of Inversion.

        constraints : .constraints.Constraints
            An object that handles the constraint conditions that wish to be
            used to constraint the inversion. The constraints will be solved
            for exactly using the method of lagrange multipliers.

        regularisation : .regularisation.Regulariser
            An object that handles the populating the regularisation matrices
            (Γ) and the regularisation coefficients (α) for all terms in the
            Equation. It is stored as a single matrix that is composed of sub-
            matrices of Γ applied to each term in the equation as they occur
            in the Equation. In other words each term has its own separate
            regulariser.

        Returns
        -------
        None or .constructors.ModelArray

        """

        g, D = G.matrix, d.array

        if regularisation is not None:

            gamma = regularisation.gamma.matrix.T \
                @ regularisation.gamma.matrix

            gamma = coo_matrix(gamma)

            self.__apply_alpha(gamma, regularisation.term_map)

            g = vstack((g, gamma))
            D = vstack((D, np.zeros((gamma.shape[0], 1))))

        GTG, GTd = compress_matrices(g, D)

        if constraints is not None:  # apply constraints from Constraints.
            F = constraints.F.matrix
            h = constraints.h.array

            GTG, GTd = apply_constraints(GTG, GTd, F, h)

        invout = lstsq(GTG.toarray(), GTd.toarray())

        m = ModelArray(G.term_map, invout[0])

        if not inplace:
            return m
        else:
            self.m = m
            self.G = G
            self.d = d
            if constraints is not None:
                self.constraints = constraints

    def forward(self, G: DesignMatrix, m: ModelArray) -> DataArray:
        return DataArray(np.dot(G.matrix.toarray(),
                         m.array.toarray()[:G.matrix.shape[1]]
                         .reshape((G.matrix.shape[1], 1))))

    def __apply_alpha(self, reg: coo_matrix, term_map: Equation):
        for term, stuff in term_map.values.items():
            # print(term)
            if stuff['regularisation']:
                alpha = stuff['regularisation']['alpha']
                reg.data[np.isin(reg.row, stuff['model_indices'])] *= alpha

    # GETTERS AND SETTERS
    @property
    def constraints(self) -> Constraints:
        return self._constraints

    @constraints.setter
    def constraints(self, cons):
        assert type(cons) is Constraints

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, nme: str):
        assert type(nme) is str, "id must be a unique string."
        self._name = nme

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, identifier: str):
        assert type(identifier) is str, "id must be a unique string."
        self._id = identifier

    @property
    def G(self) -> DesignMatrix:
        return self._G

    @G.setter
    def G(self, g):
        assert type(g) is DesignMatrix,\
            f"G must be type {type(DesignMatrix)} not {type(g)}."
        self._G = g

    @property
    def d(self) -> DataArray:
        return self._d

    @d.setter
    def d(self, data: DataArray):
        assert type(data) is DataArray,\
            f"G must be type {type(DataArray)} not {type(data)}."
        self._d = data

    @property
    def m(self) -> ModelArray:
        return self._m

    @m.setter
    def m(self, model: ModelArray):
        assert type(model) is ModelArray,\
            f"G must be type {type(ModelArray)} not {type(model)}."
        self._m = model
