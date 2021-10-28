# da-fits

Read/write FITS files to/from Dask Arrays. Currently a very simple implementation.

NOTE: Astropy can't write from a Dask array to FITS in parallel (see [#11159](https://github.com/astropy/astropy/issues/11159)). As a workaround, I write to a temporary Zarr file (which supports parallel write), and then copy to a FITS file in serial. Keep this in mind for very large files!

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
import dafits

# See doctstring
help(dafits.read)
# Help on function read in module dafits.core:

# read(file: str, ext=0, memmap=True, mode='denywrite', chunks='auto', return_header=False) -> Tuple[dask.array.core.Array, Optional[Dict]]
#     Read FITS file to DataArray.
    
#     Args:
#         file (str): FITS file to read.
#         ext (int, optional): FITS extension to read. Defaults to 0.
#         memmap (bool, optional): Use memmap. Defaults to True.
#         mode (str, optional): Read mode. Defaults to "denywrite".
#         chunks (str, optional): Dask array chunks. Defaults to "auto".
#         return_header (bool, optional): Optionally return the FITS header. Defaults to False.
    
#     Returns:
#         typing.Tuple[da.Array, typing.Optional[typing.Dict]]: DataArray and (optionally) FITS header.

# Read a file with header
data, header = dafits.read('/path/to/file.fits', return_header=True)

# Get data in memory
data.compute()

# Do some kind of maths
new_data = data.mean(axis=0)

# Write to disk (via Zarr)
# See doctstring
help(dafits.write)
# Help on function write in module dafits.core:

# write(file: str, data: dask.array.core.Array, header=None, verbose=True, **kwargs) -> None
#     Write DataArray to FITS file (via Zarr).
    
#     Args:
#         file (str): Output filename.
#         data (da.Array): Input data.
#         header (header, optional): FITS header. Defaults to None.
#         verbose (bool, optional): Verbose output. Defaults to True.
#         **kwargs: Additional keyword arguments passed onto fits.writeto.


dafits.write('/path/to/new_file.fits', new_data, header=header, overwrite=True)
```
