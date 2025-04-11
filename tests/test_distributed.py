from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from astropy import log as logger
from astropy.io import fits
from dafits import getdata, getheader, writeto
from distributed import Client, LocalCluster

from .test_local import fits_file_3d  # noqa: F401


@pytest.fixture(scope="session")
def cluster() -> Generator[LocalCluster, None]:
    """Fixture to create a local cluster."""
    cluster = LocalCluster(n_workers=2)
    yield cluster
    cluster.close()


@pytest.mark.filterwarnings("ignore: File may have been truncated")
def test_astropy_write(fits_file_3d: Path, tmpdir: Path):  # noqa: F811
    data_da = getdata(fits_file_3d)
    header = getheader(fits_file_3d)
    output_file = Path(tmpdir) / "da_output_3d.fits"

    # works
    fits.writeto(
        output_file,
        data_da,
        header,
        overwrite=True,
    )


@pytest.mark.filterwarnings("ignore: File may have been truncated")
def test_astropy_write_distributed(
    cluster: LocalCluster,
    fits_file_3d: Path,  # noqa: F811
    tmpdir: Path,
):
    logger.info(f"{cluster=}")
    data_da = getdata(fits_file_3d)
    header = getheader(fits_file_3d)
    output_file = Path(tmpdir) / "da_output_3d.fits"

    fits.writeto(
        output_file,
        data_da,
        header,
        overwrite=True,
    )

    with Client(cluster) as client:
        logger.info(f"{client=}")
        writeto(
            filename=output_file,
            data=data_da,
            header=header,
            overwrite=True,
        )
