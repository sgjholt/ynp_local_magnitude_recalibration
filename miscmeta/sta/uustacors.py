"""A collection of functions/classes to parse UUSS station corrections file.

This file can also be imported as a module and contains the following
classes:

    * X
        - Y

Note: Indentation means it is a sub-class of the object in the next highest
scope.
"""
import os
import numpy as np
from typing import Union, Callable

# PATH NAME TO HERE
PDIR = os.path.dirname(os.path.abspath(__file__))
UUSTACORS = "data/sta_cors_P07.txt"

