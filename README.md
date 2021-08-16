# da-fits

Read FITS files into Dask Arrays. Currently a very simple implementation.

Inspired by:
- https://github.com/sunpy/sunpy/issues/2715
- https://github.com/ska-sa/xarray-fits

## Installation

Install from PyPi
```
pip install dafits
```
Or, from GitHub:
```
pip install git+https://github.com/AlecThomson/da-fits.git
```

## Example usage

```python
from dafits import DaFits

# See doctstring
help(DaFits)

# Use astropy
da_fits = DaFits('/path/to/file.fits', use_fitsio=False, memmap=True, mode='denywrite')

# Use FITSIO -- can be faster than astropy
da_fits = DaFits('/path/to/file.fits', hdu=0, chunks='auto')

# Access dask array
data = da_fits.data

# Access header
header = da_fits.header

# Get data in memory
data.compute()
```
