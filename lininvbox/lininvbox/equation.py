"""Design Matrix Constructor Functions

This file contains the classes

This file can also be imported as a module and contains the following
classes:

    * Equation

"""

# TODO: Consider making a param maps super class and have param map be a
# subclass that returns a Equation class when they're added together.
# Alternately, you can make all variables optional in the constructor and just
# pass an updated dictionary to it.

import numpy as np
from typing import Union
from copy import deepcopy
from collections import OrderedDict
from .utils import build_base_term_map, pmap_checklist

PMAP_RAW = build_base_term_map()
POUT_MAX = 3
SUPPORTED_TERMS = ['CONSTANT', 'LINEAR INTERPOLATION']


class Equation():

    """
    A class used as a container that is initialised for a group of Terms.
    Equations are created by 'adding' multiple Term instances together.


    ...

    Sub-classes
    -----------
        Term
            A class that relates an equation term with the data to assist
            with the translation to Matrix representation.

    Attributes
    ----------
    values : OrderedDict(
                    name = OrderedDict(
                        indexes = np.ndarray,
                        labels = np.ndarray,
                        values = np.ndarray,
                        )
                    )
        A dictionary which tracks and stores the equation term data label and
        index metadata.

    Methods
    -------
    modify_sign(name: str, sign: int)
        Edits the sign of the equation term to a value of 1 (positive term) or
        -1 (negative term).

    get_term(name: str)
        Retrieves the term from the param_map dictionary.

    change_term_name(oldname: str, newname: str)
        Changes the name of a term.
    """

    def __init__(self, term_values: Union[OrderedDict, None] = None):
        if term_values is not None:
            self.values = term_values
        else:
            self.values = deepcopy(PMAP_RAW)

        self.__map_model_indices()

    def __map_model_indices(self) -> None:

        try:
            tm = self.values
            lastmax = -1  # set to -1 to cancel out the + 1 for first iteration
            for group in tm.keys():
                tm[group]
                tm[group]['model_indices'] = tm[group]['unique_indices']\
                    + lastmax + 1

                lastmax = int(tm[group]['model_indices'].max())

            self.npars = lastmax + 1

        except ValueError:
            pass

    def change_term_name(self, oldname: str, newname: str) -> None:
        """
        Changes a group name in the parameter map.

        ...

        Parameters
        ----------

        oldname : str
            The current name of the term to be changed.

        newname : str
            The new term name.
        """

        assert type(newname) is str and type(oldname) is str,\
            "name variables must be str and a valid parameter map name."
        self.values[newname] = self.values.pop(oldname)

    def modify_sign(self, name: str, sign: int) -> None:
        """
        Parameters
        ----------
        name : str
            A unique name of the parameter to be mapped, e.g. Dist.

        sign : int [1, -1]
            The sign of the equation, defaults to 1 (positive sign).
        """
        assert sign == 1 or sign == -1,\
            "The variable sign must be int value of 1 or -1."
        self.values[name]['sign'] = np.array([sign], dtype=int)

    def get_term(self, name: str) -> np.ndarray:
        """
        Gets a parameter map group using the group name key.
        ...

        Parameters
        ----------

        name : str
            The name of the group to be retrieved from the parameter map dict.
        """

        assert type(name) is str,\
            "name variable must be str and a valid parameter map name."
        return self.values[name]

    def __add__(self, other):

        # Make copies of the pmaps so the originals are not modified
        this_pmap, other_pmap = deepcopy(self.values), deepcopy(other.values)
        this_pmap.update(other_pmap)  # update the first copy with the second
        # return a new Equation isntance with the updated term-value mappings.
        return Equation(this_pmap)

    def __repr__(self):
        try:
            sout = """ """
            # sout += f"Model length : {self.npars}\n"
            for key in self.values.keys():
                sout += (str(key) + "\n")
                sout += ("-" * int(len(key) + 2) + "\n")
                for skey in self.values[key]:
                    out = self.values[key][skey]
                    if type(out) is list or type(out) is np.ndarray:
                        if len(out) >= POUT_MAX:
                            sout += f"{skey} : {out[:POUT_MAX]}...{out[-POUT_MAX:]}\n"
                    else:
                        sout += f"{skey} : {out}\n"

                sout += ("-" * 80 + "\n")

            return sout
        except AttributeError:
            return "EMPTY"

    def __str__(self):
        return self.__repr__()

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        pmap_checklist(values)
        self._values = values

    @property
    def npars(self) -> int:
        return self._npars

    @npars.setter
    def npars(self, num: int):
        assert type(num) is int, "npars must be an integer."
        self._npars = num


class Term(Equation):
    """
    A class used as a container that is initialised for a single
    term. Terms are designed to be passed into design matrix constructors to
    allocate coefficients to a matrix or an array. Terms can be 'added'
    together and are returned Equation instance.

    ...

    Attributes
    ----------
    See Equation.

    Methods
    -------
    See Equation.

    Private Methods
    ---------------
    __populate_term_values()
        Moves parameter metadata to the parameter dictionary.
    """

    def __init__(self,
                 name: str,
                 kind: str,
                 labels: np.ndarray,
                 unique_labels: Union[np.ndarray, None] = None,
                 sign: int = 1
                 ):
        """
        Parameters
        ----------
        name : str
            A unique name of the parameter to be mapped, e.g. Dist or None.
        kind : str
            A unique name of the parameter to be mapped, e.g. Dist or None.
        labels : np.ndarray
            The parameter labels (e.g. np.array([1, 10, 100, 1000])) or None.
        unique_labels : np.ndarray, None
            The unique entries in labels or None.
        sign : int [1 or -1]
            The sign of the parameters in the equation.


        Raises
        ------
        AssertionError
            Checks parameters for expected types.
            The correct type is recommended if it is raised.
        """

        super().__init__()

        # Check to make sure basic type requirements are met, raise assertion
        # error otherwise. Make custom errors later?
        assert type(name) is str,\
            f"The variable 'name' must be type str not {type(name)}"
        assert type(kind) is str,\
            f"The variable 'kind' must be type str not {type(labels)}"
        assert type(labels) is np.ndarray,\
            "The variable 'labels' must be type np.ndarray not" +\
            f"{type(labels)}"

        # Since this is optional we need another check. We should give
        # people a chance to sort their unique entries themselves if they
        # have tricky data.
        if unique_labels is None:  # assume it will be None
            unique_labels = np.unique(labels)
        else:  # if not just check the data type is right
            assert type(unique_labels) is np.ndarray,\
                f"The variable {unique_labels} must be type" +\
                f"np.ndarray not {type(unique_labels)}"

        unique_indices = np.indices(unique_labels.shape).flatten()

        self.values = self.__populate_term_values(name,
                                                  kind,
                                                  unique_indices,
                                                  unique_labels,
                                                  labels,
                                                  )
        self.modify_sign(name, sign)

    def __populate_term_values(self,
                               name: str,
                               kind: str,
                               unique_indices: np.ndarray,
                               unique_labels: np.ndarray,
                               raw_labels: np.ndarray,
                               ) -> OrderedDict:
        """
        Parameters
        ----------
        name : str
            A unique name of the parameter term to be mapped, e.g. Dist.
        unique_indices : np.ndarray
            The parameter labels (e.g. np.array([1, 10, 100, 1000])).
        unique_labels : np.ndarray
            The unique entries in labels
        raw_labels : np.ndarray
            All entries of labels.

        Returns: OrderedDict(OrderedDict)

        """
        # make a deepcopy of the architype so that each instance of the class
        # points to a different values in memory.
        values = deepcopy(PMAP_RAW)

        values[name] = values.pop("name")
        values[name]['kind'] = kind
        values[name]['unique_indices'] = unique_indices.flatten()
        values[name]['unique_labels'] = unique_labels.flatten()
        values[name]['raw_labels'] = raw_labels.flatten()

        return values
