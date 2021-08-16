#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dask.array as da

class DaFits:
    def __init__(self, file: str, ext=0, chunks='auto', use_fitsio=True, memmap=True, mode='denywrite'):
        """Extract FITS data as Dask 

        Args:
            file (str): Name of FITS file.
            ext (int, optional): HDU extension to select. Defaults to 0.
            chunks (int, tuple, optional):
                How to chunk the array. Must be one of the following forms:

                - A blocksize like 1000.
                - A blockshape like (1000, 1000).
                - Explicit sizes of all blocks along all dimensions like
                ((1000, 1000, 500), (400, 400)).
                - A size in bytes, like "100 MiB" which will choose a uniform
                block-like shape
                - The word "auto" which acts like the above, but uses a configuration
                value ``array.chunk-size`` for the chunk size

                -1 or None as a blocksize indicate the size of the corresponding
                dimension.
            use_fitsio (bool, optional): Use FITSIO for IO. Defaults to True. Otherwise uses astropy.io.
            memmap (bool, optional): For astropy,fts. Defaults to True.
            mode (str, optional): For astropy.io.fits. Defaults to 'denywrite'.
        Attributes:
            data (dask.array.Array): Dask Array wrapper around data.
            header (header): FITS header. Read by either astropy or FITSIO.
        """        
        self.file = file
        self.chunks = chunks
        self.ext = ext
        self.memmap = memmap
        self.mode = mode

        if use_fitsio:
            data, header = self.read_fitsio()
        else:
            data, header = self.read_fits()

        self.data = data
        self.header = header

    def read_fits(self):
        from astropy.io import fits
        with fits.open(self.file, memmap=self.memmap, mode=self.mode) as hdul:
            hdu = hdul[self.ext]
            data = hdu.data
            header = hdu.header
        array = da.from_array(data, chunks=self.chunks)
        return array, header

    def read_fitsio(self):
        from fitsio import FITS,FITSHDR
        with FITS(self.file) as hdul:
            hdu = hdul[self.ext]
            data = hdu.read()
            header = hdu.read_header()
        array = da.from_array(data, chunks=self.chunks)
        return array, header