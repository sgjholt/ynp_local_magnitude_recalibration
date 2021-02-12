"""CleanAmps.py is a module that contains functions that
are designed to help clean a pandas dataframe of 'amplitude level' issues.
The 'amplitude level' issues that are relatated to physical conditions
are outlined in the Holt et al. (in prep) paper. There are other functions that are
cleaning issues in formatting of the data table which are note mentioned in the
manuscript but are described here."""

## TODO: Refactor functions to put constant values as default arguments
## of the functions. This will allow the user to experiement with hard
## values in the notebook or script of their choosing without messing
## with the constants module.

import numpy as np
import pandas as pd
from typing import Union
from .constants import CleanConstants
from obspy import Inventory, read_inventory

C = CleanConstants()
## Funcs that make adjustments to existing data in the DataFrame
def fill_legacy_snr(df:pd.DataFrame,
        snr_pad: Union[int, float]=C.LEGACY_PAD_SNR) -> None:
    """
    The legacy data has no SNR information.
    This will pad the value to 2 if it is NaN.

    Returns: None
    """
    df.loc[df.Evid.astype(str).str.startswith("5"), ['SnrE', 'SnrN']] = snr_pad

def convert_cm_to_mm(df:pd.DataFrame) -> None:
    """
    Multiply any row amplitude entry corresponding to cm by 10
    to get mm. Then convert the cm entry to mm

    Returns None
    """
    for en in ['E','N']:
        df.loc[df[f"Un{en}"] == "cm", f"Amp{en}"] *= 10
        df.loc[df[f"Un{en}"] == "cm", f"Un{en}"] = "mm"

def clean_comp_cols(df:pd.DataFrame) -> None:
    """
    There is a small issue in the table where the 'Cmp' component
    column has a direction character. The last one isn't needed
    since the amps for N and E are labeled in separate columns
    in the same row.

    Returns: None
    """
    df["Cmp"] = df["Cmp"].str[:-1]

def adjust_depth_to_average_elevation(df: pd.DataFrame) -> None:
    """
    The catalog depth is provided relative to sea level. Some
    computations require that the catalog depth is corrected to
    some average surface elevation (2 km) in Yellowstone.

    Returns: None
    """
    df['EqDep'] += C.AVERAGE_SURFACE_ELEVATION

## Funcs that remove 'bad' data based on arbitrary conditions
# some perform inplace operations and others that might require closer
# attention return a modified copy.
def remove_nan_amp_rows(df: pd.DataFrame) -> None:
    """
    Can only use rows that have a N and E p-p amplitude,
    therefore rows without both are not useful.

    Returns: None
    """
    df.dropna(subset=['AmpE', 'AmpN'], inplace=True)
    df.reset_index(drop=True, inplace=True)

def remove_snr_below_thresh(df:pd.DataFrame) -> None:
    """
    UUSS Only uses amplitudes that have an SNR > 2.
    This functions find the dataframe indices of
    rows that have an SNR < 2 on either the N or E
    component and removes the entire row entry. Our
    logic being that we require both components to
    have a high SNR. Might consider requiring the
    SNRs to be similar within a tolerance in the
    future.

    Returns: None
    """
    SNR = C.MINIMUM_SNR
    df.drop(((df.loc[df['SnrE'] < SNR].index)&(df.loc[df['SnrN'] < SNR].index)), inplace=True)
    df.reset_index(drop=True, inplace=True)

def remove_near_amps_w_bad_focal_dep(df:pd.DataFrame) -> pd.DataFrame:
    """
    At a certain epicentral distance the corresponding hypocentral
    distance will become insensitive to focal depth. So we can keep
    amplitudes that are far enough away from the hypocenter where
    it doesn't matter if the depth is well-constrained or not.
    This sensitivity computation was performed by Jim Pechmann and
    is explained in the electronic supplement to Holt et al. (in prep).

    Returns: pd.DataFrame
    """
    bad_amps = df[(df['Repi']<C.MIN_REPI_BAD_DEP)&~df['GoodDep']].index
    df = df.drop(bad_amps).reset_index(drop=True)

    return df


def cut_outlier_amps_on_difference(df:pd.DataFrame, diff:float) -> pd.DataFrame:
    """
    Cuts out 'outlier' amplitudes based on if the their log-
    difference is >= a given value. Likely, this
    value has some basis with the standard
    deviation (s) of the differences. E.g. 4*s.

    This should be justified somehow. In our case a log-difference
    > 4s might likely be caused by one of the components being
    'bad' when the measurement was taken. Value should be picked with
    caution. Returns a DataFrame with rows containing 'bad' amplitudes
    removed and the dataframe index is reset.

    Returns: pd.DataFrame
    """

    E = df['AmpE'].apply(np.log10)
    N = df['AmpN'].apply(np.log10)

    dpp = N-E

    return df[np.abs(dpp)<=diff].reset_index(drop=True)

def cut_events_below_min_amp_count_thresh(df:pd.DataFrame) -> pd.DataFrame:
    """
    At a certain epicentral distance the corresponding hypocentral
    distance will become insensitive to focal depth. So we can keep
    amplitudes that are far enough away from the hypocenter where
    it doesn't matter if the depth is well-constrained or not.
    This sensitivity computation was performed by Jim Pechmann and
    is explained in the electronic supplement to Holt et al. (in prep).

    Returns: pd.DataFrame
    """
    ev_count = df.groupby('Evid').size()
    accept_evs = ev_count[ev_count >= C.MIN_AMPS_FOR_EV]
    df = df[df.Evid.isin(accept_evs.index.values)]
    df.reset_index(drop=True)

    return df

def reduce_to_single_instrument_per_event(df:pd.DataFrame) -> None:
    """
    The UUSS has composite stations with multiple sensors (e.g.,
    broadband, strong motion and short period) at one location. They
    routinely pick p-p ampltiudes on strong motion and broadband stations.
    Therefore, to not bias a site we must choose which instrument and associated
    p-p amplitude to keep so we have only one per station per event. Here,
    we prefer the p-p measurement with the highest signal-to-noise ratio (SNR)
    on both horizontal components. Usually, this will be on the broadband sensor
    but for large earthquakes this might not be the case.
    """

    df.sort_values(['Evid', "Sta", "Cmp", "SnrE", "SnrN"], inplace=True)
    df.drop_duplicates(subset=["Evid", "Sta"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)


def remove_mixed_station_meta_rows(df: pd.DataFrame,
                                   ) -> None:

    inv = read_inventory("../miscmeta/sta/StationMetaData.xml")

    bad_inds = []
    for i, row in df.iterrows():
        meta = None
        seeds = [f"{row.Net}.{row.Sta}.{loc}.{row.Cmp}N" for loc in [
            "", "01", "02", "10", "20"]]
        for seed in seeds:
            try:
                meta = inv.get_coordinates(seed)
                break
            except Exception:
                continue
        if meta is None:
            print(f"We got a problem with {seed}.")

        if np.abs(row.SLat - meta["latitude"]) >= 0.001:
            print(seed)
    #         print(row.SLat - meta["latitude"])
            bad_inds.append(i)

    df.drop(bad_inds, inplace=True)
    df.reset_index(drop=True, inplace=True)

## Funcs that create new columns from existing information.

def create_mean_half_pp_amp_col(df:pd.DataFrame) -> None:
    """
    The amplitudes are defined as peak-to-peak (p-p) in mm (or cm if not corrected).
    For this project and for UUSS continuity we are defining the distance correction
    on the average horizontal *half* p-p amplitude. We can do both steps (average and half)
    in one go by adding the amplitudes together and dividing by 4.

    Returns: None
    """
    df["halfAmpH"] = (df["AmpE"] + df["AmpN"])/4

def create_catalog_magnitude_col(df: pd.DataFrame) -> None:
    """
    The UUSS prefers ML over MC when available, the catalog magnitude
    would be ML and then MC if no ML was assigned. This function
    creates a new column that simply shows the catalog magnitude.
    -9.99 is a flag signalling no magnitude was assigned.
    """
    df['CatMag'] = np.where(df['EqML'] == -9.99, df['EqMC'], df['EqML'])

def create_focal_dep_dmin_ratio_col(df: pd.DataFrame) -> None:
    """
    The UUSS prefers ML over MC when available, the catalog magnitude
    would be ML and then MC if no ML was assigned. This function
    creates a new column that simply shows the catalog magnitude.
    -9.99 is a flag signalling no magnitude was assigned.

    Returns: None
    """
    df['H'] = np.abs(df['Dmin']/df['EqDep'])


def add_focal_depth_quality_col(df: pd.DataFrame) -> None:
    """
    We have a specific set of criteria to determine if a given event's
    focal depth is well-constrained or not. We reqiuire that the ratio
    of minimimum recording distance (Dmin [km]) and focal depth (h [km]), H,
    is <= some threshold or Dmin itself is <= some close distance for shallow
    events. We also require that the maximum vertical error (ErrZ [km]) is no
    greater than a threshold amount. If the ErrZ is larger than the threshold
    it automatically is set to False even if either of the other two conditions
    are satisfied.

    Returns: None
    """
    query = np.where(((
        df['H']<=C.H_MINIMUM) | (df['Dmin']<=C.DMIN_MINIMUM)) & (
        df['ErrZ'] <= C.ERZ_MINIMUM), True, False)

    df['GoodDep'] = query
