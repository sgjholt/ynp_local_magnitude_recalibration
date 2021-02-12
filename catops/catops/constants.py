"""
This module contains a set of constant values that are relevent to formatting
and cleaning the database.
"""


class CleanConstants():

    LEGACY_PAD_SNR = 2
    MINIMUM_SNR = 2
    AVERAGE_SURFACE_ELEVATION = 2
    H_MINIMUM = 1.4
    DMIN_MINIMUM = 5
    ERZ_MINIMUM = 2
    MIN_REPI_BAD_DEP = 50
    MIN_AMPS_FOR_EV = 2

    def __init__(self):
        pass


# reduce in lat lon and dep
class GeoBalanceConstants():

    LON_MIN = -113.5
    LON_MAX = -109
    LAT_MIN = 43.7
    LAT_MAX = 45.7
    DEP_MIN = 0
    DEP_MAX = 25
    LAT_SLICES = 35
    LON_SLICES = 35
    DEP_SLICES = 6
    EV_MAX = 10

    def __init__(self):
        pass


# reduce in lat lon only

# class GeoBalanceConstants():

#     LON_MIN = -113.5
#     LON_MAX = -109
#     LAT_MIN = 43.7
#     LAT_MAX = 45.7
#     DEP_MIN = 0
#     DEP_MAX = 25
#     LAT_SLICES = 55
#     LON_SLICES = 35
#     DEP_SLICES = 2
#     EV_MAX = 15

#     def __init__(self):
#         pass
