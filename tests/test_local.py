from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from dafits import getdata, getheader, writeto


def create_nd_fits_file(
    array_shape: tuple[int, ...],
    filename: Path,
) -> Path:
    total_size = np.prod(array_shape)
    data = np.arange(total_size, dtype=np.float64).reshape(array_shape)
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -64
    header["NAXIS"] = len(array_shape)
    for i, size in enumerate(array_shape):
        header[f"NAXIS{i + 1}"] = size

    fits.writeto(
        filename,
        data,
        header=header,
        overwrite=True,
    )
    return filename


@pytest.fixture
def fits_file_1d(tmpdir) -> Generator[Path, None]:
    """Fixture to create a 1D array."""

    fits_path = Path(tmpdir) / "oned.fits"

    create_nd_fits_file(
        array_shape=(2,),
        filename=fits_path,
    )

    yield fits_path

    fits_path.unlink()


@pytest.fixture
def fits_file_2d(tmpdir) -> Generator[Path, None]:
    """Fixture to create a 2D array."""

    fits_path = Path(tmpdir) / "twod.fits"

    create_nd_fits_file(
        array_shape=(1, 2),
        filename=fits_path,
    )

    yield fits_path

    fits_path.unlink()


@pytest.fixture
def fits_file_3d(tmpdir) -> Generator[Path, None]:
    """Fixture to create a 3D array."""

    fits_path = Path(tmpdir) / "threed.fits"

    create_nd_fits_file(
        array_shape=(1, 2, 3),
        filename=fits_path,
    )

    yield fits_path

    fits_path.unlink()


@pytest.fixture
def fits_file_4d(tmpdir) -> Generator[Path, None]:
    """Fixture to create a 4D array."""

    fits_path = Path(tmpdir) / "fourth.fits"

    create_nd_fits_file(
        array_shape=(1, 2, 3, 4),
        filename=fits_path,
    )

    yield fits_path

    fits_path.unlink()


@pytest.mark.filterwarnings("ignore: File may have been truncated")
def test_read_1d(fits_file_1d: Path):
    """Test reading a 1D FITS file."""

    data_da = getdata(fits_file_1d)
    data_np = fits.getdata(fits_file_1d)

    assert data_da.shape == data_np.shape
    assert data_da.dtype == data_np.dtype
    assert np.array_equal(data_da.compute(), data_np)


@pytest.mark.filterwarnings("ignore: File may have been truncated")
def test_write_1d(fits_file_1d: Path, tmpdir: Path):
    """Test writing a 1D FITS file."""

    data_da = getdata(fits_file_1d)
    header = getheader(fits_file_1d)
    output_file = Path(tmpdir) / "da_output_1d.fits"

    writeto(filename=output_file, data=data_da, header=header, overwrite=True)

    data_np_input = fits.getdata(fits_file_1d)
    data_np_output = fits.getdata(output_file)

    assert data_np_input.shape == data_np_output.shape
    assert data_np_input.dtype == data_np_output.dtype
    assert np.array_equal(data_np_input, data_np_output)

    output_file.unlink()


@pytest.mark.filterwarnings("ignore: File may have been truncated")
def test_read_2d(fits_file_2d: Path):
    """Test reading a 2D FITS file."""

    data_da = getdata(fits_file_2d)
    data_np = fits.getdata(fits_file_2d)

    assert data_da.shape == data_np.shape
    assert data_da.dtype == data_np.dtype
    assert np.array_equal(data_da.compute(), data_np)


@pytest.mark.filterwarnings("ignore: File may have been truncated")
def test_write_2d(fits_file_2d: Path, tmpdir: Path):
    """Test writing a 2D FITS file."""

    data_da = getdata(fits_file_2d)
    header = getheader(fits_file_2d)
    output_file = Path(tmpdir) / "da_output_2d.fits"

    writeto(filename=output_file, data=data_da, header=header, overwrite=True)

    data_np_input = fits.getdata(fits_file_2d)
    data_np_output = fits.getdata(output_file)

    assert data_np_input.shape == data_np_output.shape
    assert data_np_input.dtype == data_np_output.dtype
    assert np.array_equal(data_np_input, data_np_output)

    output_file.unlink()


@pytest.mark.filterwarnings("ignore: File may have been truncated")
def test_read_3d(fits_file_3d: Path):
    """Test reading a 3D FITS file."""

    data_da = getdata(fits_file_3d)
    data_np = fits.getdata(fits_file_3d)

    assert data_da.shape == data_np.shape
    assert data_da.dtype == data_np.dtype
    assert np.array_equal(data_da.compute(), data_np)


@pytest.mark.filterwarnings("ignore: File may have been truncated")
def test_write_3d(fits_file_3d: Path, tmpdir: Path):
    """Test writing a 3D FITS file."""

    data_da = getdata(fits_file_3d)
    header = getheader(fits_file_3d)
    output_file = Path(tmpdir) / "da_output_3d.fits"

    writeto(filename=output_file, data=data_da, header=header, overwrite=True)

    data_np_input = fits.getdata(fits_file_3d)
    data_np_output = fits.getdata(output_file)

    assert data_np_input.shape == data_np_output.shape
    assert data_np_input.dtype == data_np_output.dtype
    assert np.array_equal(data_np_input, data_np_output)

    output_file.unlink()


@pytest.mark.filterwarnings("ignore: File may have been truncated")
def test_read_4d(fits_file_4d: Path):
    """Test reading a 4D FITS file."""

    data_da = getdata(fits_file_4d)
    data_np = fits.getdata(fits_file_4d)

    assert data_da.shape == data_np.shape
    assert data_da.dtype == data_np.dtype
    assert np.array_equal(data_da.compute(), data_np)


@pytest.mark.filterwarnings("ignore: File may have been truncated")
def test_write_4d(fits_file_4d: Path, tmpdir: Path):
    """Test writing a 4D FITS file."""

    data_da = getdata(fits_file_4d)
    header = getheader(fits_file_4d)
    output_file = Path(tmpdir) / "da_output_4d.fits"

    writeto(filename=output_file, data=data_da, header=header, overwrite=True)

    data_np_input = fits.getdata(fits_file_4d)
    data_np_output = fits.getdata(output_file)

    assert data_np_input.shape == data_np_output.shape
    assert data_np_input.dtype == data_np_output.dtype
    assert np.array_equal(data_np_input, data_np_output)

    output_file.unlink()
