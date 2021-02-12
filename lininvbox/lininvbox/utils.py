"""
Utils

This file contains a set of 'stand alone' utility functions for lininvbox.

This file can also be imported as a module and contains the following
functions:

    * build_base_term_map()
        Utility function to create a template for a parameter map used
        in all Matrix class and sub-class instances. See '.basetypes' for
        more detail.

    * pmap_checklist()
        A utility function that checks an arbitrary parameter map against the
        base template from build_base_term_map to see if it has the expected
        fields.

"""
import numpy as np
from datetime import datetime
from collections import OrderedDict


def build_base_term_map() -> OrderedDict:
    """
    A function that creates a basic for a parameter map that will be populated
    with parameter labels, values and index information.

    Notes
    -----
    name : str
        A unique identifier (name) of the parameter to be mapped, e.g. Dist.
    unique_indices : np.ndarray
        The parameter labels (e.g. np.array([1, 10, 100, 1000])).
    unique_labels : np.ndarray
        The unique entries in labels
    raw_labels : np.ndarray
        All entries of labels.
    model_values : np.ndarray
        The recovered values for the parameters out of the inversion.
    sign : int [1, -1]
        The sign of the equation, defaults to 1 (positive sign).

    """
    return OrderedDict(name=OrderedDict(unique_indices=np.array([]),
                                        unique_labels=np.array([]),
                                        raw_labels=np.array([]),
                                        model_indices=np.array([]),
                                        model_values=np.array([]),
                                        model_residuals=np.array([]),
                                        sign=np.array([1, ], dtype=int),
                                        kind="",
                                        constraints={},
                                        regularisation={},
                                        ),
                       )


def pmap_checklist(pmap: OrderedDict) -> None:
    """
    A function that checks to see if the parameter map is has the expected
    format. See 'build_base_term_map' for additional details.

    Parameters
    ----------
    pmap : OrderedDict(OrderedDict)
        A parameter map to be checked against the expected format.

    """

    PMAP_RAW = build_base_term_map()

    for key in pmap.keys():
        assert pmap[key].keys() == PMAP_RAW["name"].keys(),\
            "Attribute: pmap keys are not as expected." +\
            "\n If set manually please ensure it is" +\
            "created using build_base_term_map in from utils."
        got = [type(value) for value in pmap[key].values()]
        expected = [type(value) for value in PMAP_RAW["name"].values()]
        assert got == expected, f"{got} : {expected}"


def get_timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
