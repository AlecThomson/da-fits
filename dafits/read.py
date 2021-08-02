#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" FITS to Dask Arrays """

import dask.array as da
from dafits.delayed import DelayedFITS, DelayedFITSIO


def da_from_fits(filename: str, hdu=0, chunks='auto') -> da.Array:
    """Read a FITS file into a Dask Array.
    Uses astropy.io for reading.

    Args:
        filename (str): Name of FITS file.
        hdu (int, optional): HDU to select. Defaults to 0.
        chunks (int, tuple, optional):
            How to chunk the array. Must be one of the following forms:

            - A blocksize like 1000.
            - A blockshape like (1000, 1000).
            - Explicit sizes of all blocks along all dimensions like
            ((1000, 1000, 500), (400, 400)).
            - A size in bytes, like "100 MiB" which will choose a uniform
            block-like shape
            - The word "auto" which acts like the above, but uses a configuration
            value ``array.chunk-size`` for the chunk size

            -1 or None as a blocksize indicate the size of the corresponding
            dimension.
    Returns:
        da.Array: Delayed read of data into Dask Array.
    """    
    array = da.from_array(DelayedFITS(filename), chunks=chunks)
    return array


def da_from_fitsio(filename: str, hdu=0, chunks='auto') -> da.Array:
    """Read a FITS file into a Dask Array.
    Uses FITSIO for reading.

    Args:
        filename (str): Name of FITS file.
        hdu (int, optional): HDU to select. Defaults to 0.
        chunks (int, tuple, optional):
            How to chunk the array. Must be one of the following forms:

            - A blocksize like 1000.
            - A blockshape like (1000, 1000).
            - Explicit sizes of all blocks along all dimensions like
            ((1000, 1000, 500), (400, 400)).
            - A size in bytes, like "100 MiB" which will choose a uniform
            block-like shape
            - The word "auto" which acts like the above, but uses a configuration
            value ``array.chunk-size`` for the chunk size

            -1 or None as a blocksize indicate the size of the corresponding
            dimension.
    Returns:
        da.Array: Delayed read of data into Dask Array.
    """   
    array = da.from_array(DelayedFITSIO(filename), chunks=chunks)
    return array
