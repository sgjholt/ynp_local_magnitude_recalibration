"""A collection of Local Magnitude Scales

This file contains the classes that read and store information related to
various local magnitudes scales.

This file can also be imported as a module and contains the following
classes:

    * MagScale
        - Richter1958

Note: Indentation means it is a sub-class of the object in the next highest
scope.
"""
import os
import numpy as np
from typing import Union, Callable

# PATH NAME TO HERE
PDIR = os.path.dirname(os.path.abspath(__file__))
RICHTER1958 = "data/richter1956.csv"


class MagScale():
    """
    Stuff.
    """

    def __init__(self,
                 equation: Union[Callable[[np.ndarray, ], np.ndarray], None] = None,
                 logA0: Union[np.ndarray, None] = None,
                 distances: Union[np.ndarray, None] = None,
                 ):

        if equation is not None:
            self.equation = equation
        if logA0 is not None:
            self.logA0 = logA0
        if distances is not None:
            self.distances = distances

    def epi_to_hypo(self, av_dep: float):
        self.distances = np.sqrt(self.distances**2 + av_dep**2)

    @property
    def equation(self) -> Callable:
        return self._equation

    @equation.setter
    def equation(self, eq: Callable):
        assert type(eq) is Callable
        self._equation = eq

    @property
    def logA0(self) -> np.ndarray:
        return self._logA0

    @logA0.setter
    def logA0(self, lA0: np.ndarray):
        assert type(lA0) is np.ndarray
        self._logA0 = lA0

    @property
    def distances(self) -> np.ndarray:
        return self._distances

    @distances.setter
    def distances(self, dists: np.ndarray):
        assert type(dists) is np.ndarray
        self._distances = dists


class Richter1958(MagScale):
    """
    The Richter 1958 Local Magnitude Scale for Southern and Central California.
    The data table was obtained from Boore (1989).


    Notes
    -----
        References
        ----------
        Boore, D.M. (1989). The Richter scale: its development and use for
            determining earthquake source parameters. Tectonophysics, 166(1-3),
            pp.1-14.
    """

    def __init__(self,
                 path: Union[str, None] = os.path.join(PDIR, RICHTER1958)
                 ):

        out = self.read_csv_file(path)
        distances, logA0 = out[:, 0], out[:, 1]

        super().__init__(logA0=logA0, distances=distances)

    def read_csv_file(self, path: str) -> np.ndarray:
        return np.loadtxt(path, delimiter=",", skiprows=1)
