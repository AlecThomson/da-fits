#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from astropy.io import fits
import dask.array as da
import shutil
import typing
import zarr
from dask import delayed
import numpy as np

def mmap_load_chunk(
    data: np.ndarray, 
    sl:slice,
    ext:int=0,
    memmap:bool=True,
    mode:str="denywrite",
) -> da.Array:
    """Create a memory-mapped array from a FITS file.

    Args:
        file (str): FITS file to read.
        sl (slice): Slice to read.

    Returns:
        da.Array: Dask array chunk.
    """
    return data[sl]

def mmap_dask_array(
    file:str,
    ext:int=0,
    memmap:bool=True,
    mode:str="denywrite",
    blocksize:int=5
) -> da.Array:
    """Memmap a FITS file to a Dask array.

    Args:
        file (str): FITS file to read.
        ext (int, optional): HDUList extension. Defaults to 0.
        memmap (bool, optional): Read using memmap. Defaults to True.
        mode (str, optional): FITS read mode. Defaults to "denywrite".
        blocksize (int, optional): hunksize along fast (last) axis. Defaults to 5.

    Returns:
        da.Array: Full Dask array.
    """
    # arr = mmap_load_chunk(
    #     file=file, 
    #     sl=slice(0,-1),
    #     ext=ext, 
    #     memmap=memmap,
    #     mode=mode,
    # )
    with fits.open(file, memmap=memmap, mode=mode) as hdulist:
        arr = hdulist[ext].data
    shape = arr.shape
    dtype = arr.dtype

    load = delayed(mmap_load_chunk)
    chunks = []
    print(shape)
    for index in range(0, shape[-1], blocksize):
        # Truncate the last chunk if necessary
        chunk_size = min(blocksize, shape[-1] - index)
        chunk = da.from_delayed(
            load(
                arr,
                sl=slice(index, index + chunk_size)
            ),
            shape= shape[:-1] + (chunk_size,),
            dtype=dtype
        )
        chunks.append(chunk)
    return da.concatenate(chunks, axis=-1)

def read(
    file: str,
    ext:int=0,
    memmap:bool=True,
    mode:str="denywrite",
    return_header: bool = False,
    blocksize:int=5,
) -> typing.Tuple[da.Array, typing.Optional[typing.Dict]]:
    """Read FITS file to DataArray.

    Args:
        file (str): FITS file to read.
        ext (int, optional): FITS extension to read. Defaults to 0.
        memmap (bool, optional): Use memmap. Defaults to True.
        mode (str, optional): Read mode. Defaults to "denywrite".
        return_header (bool, optional): Optionally return the FITS header. Defaults to False.
        blocksize (int, optional): Chunksize along fast (last) axis. Defaults to 5.

    Returns:
        typing.Tuple[da.Array, typing.Optional[typing.Dict]]: DataArray and (optionally) FITS header.
    """
    # Distribute the read using memmap
    array = mmap_dask_array(
        file=file,
        ext=ext,
        memmap=memmap,
        mode=mode,
        blocksize=blocksize,        
    )

    if return_header:
        with fits.open(file, memmap=memmap, mode=mode) as hdulist:
            header = hdulist[ext].header
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
