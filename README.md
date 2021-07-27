# da-fits

Read FITS files into Dask Arrays. Currently a very simple implementation.

Inspired by:
- https://github.com/sunpy/sunpy/issues/2715
- https://github.com/ska-sa/xarray-fits

## Example usage

```python
import dafits

# Use astropy
dafits.da_from_fits('/path/to/file.fits')

# Use FITSIO -- can be faster than astropy
dafits.da_from_fitsio('/path/to/file.fits')
```
