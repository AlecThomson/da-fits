"""FITS I/O with dask."""

from __future__ import annotations

from dafits.io import getdata, getheader, writeto

from ._version import version as __version__

__all__ = ["__version__", "getdata", "getheader", "writeto"]
