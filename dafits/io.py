from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import dask.array as da
import numpy as np
from astropy import log as logger
from astropy.io import fits
from astropy.wcs import WCS
from dask import compute, delayed
from numpy.typing import ArrayLike

logger.setLevel("DEBUG")

BIT_DICT = {
    64: 8,
    32: 4,
    16: 2,
    8: 1,
}


def getdata(
    filename: Path | str,
    ext: int = 0,
) -> da.Array:
    header = fits.getheader(filename, ext=ext)
    wcs = WCS(header)
    chunks = list(wcs.array_shape)
    if len(chunks) > 2:
        chunks[0] = 1

    return da.from_array(
        fits.getdata(
            filename,
            ext=ext,
            memmap=True,
        ),
        chunks=chunks,
    )


def getheader(filename: Path | str, ext: int = 0) -> fits.Header:
    return fits.getheader(filename, ext=ext)


def _touch_header(
    output_file: Path | str,
    output_shape: tuple[int, ...],
    output_header: fits.Header,
    overwrite: bool = False,
) -> fits.Header:
    output_file = Path(output_file)
    small_size = [1 for _ in output_shape]
    data = np.zeros(small_size)
    hdu = fits.PrimaryHDU(data)
    header = hdu.header

    for key, value in output_header.items():
        header[key] = value

    header.tofile(output_file, overwrite=overwrite)
    return fits.getheader(output_file)


class FitsInit(NamedTuple):
    header: fits.Header
    dtype: np.dtype


def _init_small_file(
    output_file: Path | str,
    output_shape: tuple[int, ...],
    output_header: fits.Header,
    overwrite: bool = False,
) -> FitsInit:
    output_file = Path(output_file)
    out_arr = np.zeros(output_shape)
    logger.info(f"{out_arr.shape=}")
    fits.writeto(output_file, out_arr, output_header, overwrite=overwrite)
    with fits.open(output_file, mode="denywrite", memmap=True) as hdu_list:
        hdu = hdu_list[0]
        data = hdu.data
        on_disk_shape = data.shape
        assert data.shape == output_shape, (
            f"Output shape {on_disk_shape} does not match header {output_shape}!"
        )
    return FitsInit(fits.getheader(output_file), data.dtype)


def _init_large_file(
    output_file: Path | str,
    output_shape: tuple[int, ...],
    output_header: fits.Header,
    overwrite: bool = False,
) -> FitsInit:
    output_file = Path(output_file)
    header = _touch_header(
        output_file=output_file,
        output_shape=output_shape,
        output_header=output_header,
        overwrite=overwrite,
    )

    bytes_per_value = BIT_DICT.get(abs(output_header["BITPIX"]), None)
    msg = f"Header BITPIX={output_header['BITPIX']}, bytes_per_value={bytes_per_value}"
    logger.info(msg)
    if bytes_per_value is None:
        msg = f"BITPIX value {output_header['BITPIX']} not recognized"
        raise ValueError(msg)

    with output_file.open("rb+") as fobj:
        # Seek past the length of the header, plus the length of the
        # Data we want to write.
        # 8 is the number of bytes per value, i.e. abs(header['BITPIX'])/8
        # (this example is assuming a 64-bit float)
        file_length = len(header.tostring()) + (np.prod(output_shape) * bytes_per_value)
        # FITS files must be a multiple of 2880 bytes long; the final -1
        # is to account for the final byte that we are about to write.
        file_length = ((file_length + 2880 - 1) // 2880) * 2880 - 1
        logger.info(f"{file_length=}")
        fobj.seek(file_length)
        fobj.write(b"\0")

    with fits.open(output_file, mode="denywrite", memmap=True) as hdu_list:
        hdu = hdu_list[0]
        data = hdu.data
        on_disk_shape = data.shape
        assert on_disk_shape == output_shape, (
            f"Output shape {on_disk_shape} does not match header {output_shape}!"
        )

    return FitsInit(fits.getheader(output_file), data.dtype)


def init_fits_image(
    output_file: Path | str,
    output_header: fits.Header,
    overwrite: bool = False,
) -> tuple[fits.Header, np.dtype]:
    output_file = Path(output_file)
    if output_file.exists() and not overwrite:
        msg = f"Output file {output_file} already exists."
        raise FileExistsError(msg)

    if output_file.exists() and overwrite:
        output_file.unlink()

    output_wcs = WCS(output_header)
    output_shape = output_wcs.array_shape
    output_shape = tuple(o for o in output_shape if o != 0)
    msg = f"Creating a new FITS file with shape {output_shape}"
    logger.info(msg)
    # If the output shape is less than 1801, we can create a blank array
    # in memory and write it to disk
    if np.prod(output_shape) < 1801:
        msg = "Output cube is small enough to create in memory"
        logger.warning(msg)
        return _init_small_file(
            output_file,
            output_shape,
            output_header,
            overwrite=overwrite,
        )

    logger.info("Output cube is too large to create in memory. Creating a blank file.")
    return _init_large_file(
        output_file,
        output_shape,
        output_header,
        overwrite=overwrite,
    )


@delayed  # type: ignore[misc]
def _write_chunk_to_file(
    chunk: ArrayLike,
    filename: Path | str,
    header: fits.Header,
    block_num: int,
) -> None:
    filename = Path(filename)

    with filename.open("r+b") as file_handle:
        seek_length = len(header.tostring()) + (block_num * chunk.nbytes)
        file_handle.seek(seek_length)
        chunk.tofile(file_handle)


def writeto(
    filename: Path | str,
    data: da.Array,
    header: fits.Header,
    overwrite: bool = False,
) -> None:
    filename = Path(filename)
    if filename.exists() and not overwrite:
        msg = f"Output file {filename} already exists."
        raise FileExistsError(msg)

    if filename.exists() and overwrite:
        filename.unlink()

    header, dtype = init_fits_image(
        filename,
        header,
        overwrite=overwrite,
    )
    if data.ndim <= 2:
        new_chunksize = data.shape
    else:
        new_chunksize = list(data.shape)
        new_chunksize[0] = 1

    data = data.rechunk(new_chunksize).astype(dtype)

    results = []
    for block_num, chunk in enumerate(data.to_delayed().flatten()):
        # write to file
        result = _write_chunk_to_file(
            chunk,
            filename,
            header,
            block_num,
        )
        results.append(result)
    compute(results)
