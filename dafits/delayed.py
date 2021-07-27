
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Delayed FITS reading """

from astropy.io import fits
from fitsio import FITS,FITSHDR

class DelayedFITS:
    """Delayed FITS read with astropy.io
    """    
    def __init__(self, file: str, hdu=0):       
        self.file = file
        self.hdu = hdu
        with fits.open(self.file, memmap=True, mode='denywrite') as hdul:
            self.shape = hdul[hdu].data.shape
            self.dtype = hdul[hdu].data.dtype
            self.ndim = len(self.shape)
    def __getitem__(self, view):
        with fits.open(self.file, memmap=True, mode='denywrite') as hdul:
            return hdul[self.hdu].section[view]

class DelayedFITSIO:
    """Delayed FITS read with FITSIO
    """    
    def __init__(self, file, hdu=0):
        self.file = file
        self.hdu = hdu

        with FITS(self.file) as hdul:
            self.shape = tuple(hdul[hdu].get_dims())
            self.dtype = hdul[hdu].read().dtype
            self.ndim = len(self.shape)
    def __getitem__(self, view):
        with FITS(self.file) as hdul:
            return hdul[self.hdu][view]