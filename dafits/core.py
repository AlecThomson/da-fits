#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.io import fits
from fitsio import FITS
import dask.array as da
import shutil
import typing
import zarr
from zarr.storage import ContainsArrayError

def read(
    file: str,
    ext=0,
    memmap=True,
    mode="denywrite",
    chunks="auto",
    use_fitsio=False,
    return_header=False,
) -> typing.Tuple[da.Array, typing.Optional[typing.Dict]]:
    """Read FITS file to DataArray.

    Args:
        file (str): FITS file to read.
        ext (int, optional): FITS extension to read. Defaults to 0.
        memmap (bool, optional): Use memmap. Defaults to True.
        mode (str, optional): Read mode. Defaults to "denywrite".
        chunks (str, optional): Dask array chunks. Defaults to "auto".
        use_fitsio (bool, optional): Use FITSIO lib for I/O. Defaults to False.
        return_header (bool, optional): Optionally return the FITS header. Defaults to False.

    Returns:
        typing.Tuple[da.Array, typing.Optional[typing.Dict]]: DataArray and (optionally) FITS header.
    """
    if use_fitsio:
        with FITS(file) as hdul:
            hdu = hdul[ext]
            data = hdu.read()
            header = hdu.read_header()
    else:

        with fits.open(file, memmap=memmap, mode=mode) as hdul:
            hdu = hdul[ext]
            data = hdu.data
            header = hdu.header
    array = da.from_array(data, chunks=chunks)

    if return_header:
        ret = (array, header)
    else:
        ret = array
    return ret


def write(file: str, data: da.Array, header=None, verbose=True, use_fitsio=False, **kwargs) -> None:
    """Write DataArray to FITS file (via Zarr).

    Args:
        file (str): Output filename.
        data (da.Array): Input data.
        header (header, optional): FITS header. Defaults to None.
        verbose (bool, optional): Verbose output. Defaults to True.
        use_fitsio (bool, optional): Use FITSIO lib for I/O. Defaults to False.
    """
    # Write to temporary file
    tmp_file, z_data = write_tmp_zarr(file, data, verbose)
    if use_fitsio:
        hdu = FITS(file, mode="rw")
        hdu.write(z_data, header=header, **kwargs)
        hdu.close()
    else:
        hdu = fits.PrimaryHDU(z_data, header=header)
        hdu.writeto(file, **kwargs)
    if verbose:
        print(f"Wrote FITS file: {file}")
    shutil.rmtree(tmp_file)


def write_tmp_zarr(file: str, data: da.Array, verbose=False) -> typing.Tuple[str, da.Array]:
    """Write DataArray to temporary Zarr file.

    Args:
        file (str): Output filename.
        data (da.Array): DataArray to write.
        verbose (bool, optional): Verbose output. Defaults to False.

    Returns:
        typing.Tuple[str, da.Array]: Temporary Zarr file and data.
    """
    tmp_file = '.' + file.replace(".fits", "_tmp.zarr")
    if verbose:
        print(f"Writing temporary zarr file: {tmp_file}")
    try:
        data.to_zarr(tmp_file)
    except ContainsArrayError:
        shutil.rmtree(tmp_file)
        data.to_zarr(tmp_file)
    z_data = zarr.open(tmp_file, mode="r")
    return tmp_file, z_data
