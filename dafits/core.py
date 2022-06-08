#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.io import fits
import dask.array as da
import shutil
import typing
import zarr

def read(
    file: str,
    ext:int=0,
    memmap:bool=True,
    mode:str="denywrite",
    chunks:str="auto",
    return_header: bool = False,
    fits_kwargs: dict = {},
    dask_kwargs: dict = {},
) -> typing.Tuple[da.Array, typing.Optional[typing.Dict]]:
    """Read FITS file to DataArray.

    Args:
        file (str): FITS file to read.
        ext (int, optional): FITS extension to read. Defaults to 0.
        memmap (bool, optional): Use memmap. Defaults to True.
        mode (str, optional): Read mode. Defaults to "denywrite".
        chunks (str, optional): Dask array chunks. Defaults to "auto".
        return_header (bool, optional): Optionally return the FITS header. Defaults to False.
        fits_kwargs (dict, optional): Additional keyword arguments passed onto fits.open. Defaults to None.
        dask_kwargs (dict, optional): Additional keyword arguments passed onto dask.from_array. Defaults to None.


    Returns:
        typing.Tuple[da.Array, typing.Optional[typing.Dict]]: DataArray and (optionally) FITS header.
    """
    with fits.open(file, memmap=memmap, mode=mode, **fits_kwargs) as hdulist:
        header = hdulist[ext].header
        data = hdulist[ext].data
    array = da.from_array(data, chunks=chunks, **dask_kwargs)

    if return_header:
        return array, header
    return array


def write(
    file: str,
    data: da.Array,
    header: fits.Header = None,
    verbose=True,
    purge=True,
    **kwargs,
) -> None:
    """Write DataArray to FITS file (via Zarr).

    Args:
        file (str): Output filename.
        data (da.Array): Input data.
        header (header, optional): FITS header. Defaults to None.
        verbose (bool, optional): Verbose output. Defaults to True.
        purge (bool, optional): Purge temporary Zarr file. Defaults to True.
        **kwargs: Additional keyword arguments passed onto fits.writeto.
    """
    # Write to temporary file
    tmp_file, z_data = write_tmp_zarr(
        file=file.replace(".fits", "_tmp.zarr"),
        data=data,
        header=header,
        verbose=verbose,
    )
    hdu = fits.PrimaryHDU(z_data, header=header)
    hdu.writeto(file, **kwargs)
    if verbose:
        print(f"Wrote FITS file: {file}")
    if purge:
        shutil.rmtree(tmp_file)
        if verbose:
            print(f"Deleted temporary zarr file: {tmp_file}")


def write_tmp_zarr(
    file: str,
    data: da.Array,
    header: fits.Header = None,
    verbose: bool = False,
    overwrite: bool = False,
) -> typing.Tuple[str, da.Array]:
    """Write DataArray to temporary Zarr file.
    Computation will begin as the data is written to the Zarr file.

    Args:
        file (str): Output filename.
        data (da.Array): DataArray to write.
        verbose (bool, optional): Verbose output. Defaults to False.
        overwrite (bool, optional): Overwrite existing file. Defaults to False.

    Returns:
        typing.Tuple[str, da.Array]: Temporary Zarr file and data.
    """
    tmp_file = file
    if verbose:
        print(f"Writing temporary zarr file: {tmp_file}")
    data.to_zarr(tmp_file, overwrite=overwrite)
    if header is not None:
        _z_data = zarr.open(tmp_file, mode="r+")
        _z_data.attrs["header"] = header.tostring()
    z_data = zarr.open(tmp_file, mode="r")
    return file, z_data
