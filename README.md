# da-fits

Provies FITS I/O with Dask Arrays.

As per Astropy issue [#11159](https://github.com/astropy/astropy/issues/11159), dask arrays cannot be written right now when using `dask.distributed`

![I'll write it myself](fine.webp "I'll do it myself")

## Installation

```console
pip install dafits
```

## Usage

```python
from distributed import Client
from dafits import getdata, getheader, writeto

with Client() as client:
    # Get a dask array from an existing file
    filename = "image.fits"
    da_array = getdata(filename)
    header = getheader(filename)

    # Write a Dask array to FITS
    # This is the magic function that does not work with
    # regular astropy.io.fits
    writeto(
        "output.fits,
        da_array,
        header
    )
```

## License

`da-fits` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
