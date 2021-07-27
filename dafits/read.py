#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" FITS to Dask Arrays """

import dask.array as da
from dafits.delayed import DelayedFITS, DelayedFITSIO


def da_from_fits(filename: str, hdu=0) -> da.Array:
    """Read a FITS file into a Dask Array.
    Uses astropy.io for reading.

    Args:
        filename (str): Name of FITS file.
        hdu (int, optional): HDU to select. Defaults to 0.

    Returns:
        da.Array: Delayed read of data into Dask Array.
    """    
    array = da.from_array(DelayedFITS(filename))
    return array


def da_from_fitsio(filename: str, hdu=0) -> da.Array:
    """Read a FITS file into a Dask Array.
    Uses FITSIO for reading.

    Args:
        filename (str): Name of FITS file.
        hdu (int, optional): HDU to select. Defaults to 0.

    Returns:
        da.Array: Delayed read of data into Dask Array.
    """   
    array = da.from_array(DelayedFITSIO(filename))
    return array
