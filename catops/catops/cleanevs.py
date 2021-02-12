"""CleanCats.py is a module that contains functions that
are designed to help clean and prepare catalogs (pandas dataframe) with
'event level' issues. The 'event level' operations that are relatated to
physical conditions are outlined in the Holt et al. paper. There are other
functions that are for formatting of the data table which are
not mentioned in the manuscript but are described here."""

import numpy as np
import pandas as pd
