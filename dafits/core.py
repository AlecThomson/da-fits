#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dask.array as da
import shutil
from dataclasses import dataclass

@dataclass
class DaFits:
    def __init__(self, data = None, header = None, chunks='auto'):
        """Class for handling FITS files as Dask Arrays.

        Args:
            data (da.Array, optional): Dask Array data. Defaults to None.
            header (header, optional): FITS header. Defaults to None.
            chunks (str, optional): Dask Array chuncks. Defaults to 'auto'.
            verbose (bool, optional): Verbose output. Defaults to False.
        """               
        self.chunks = chunks
        self.data = data
        self.header = header

    def read_fits(self, file: str, ext=0, memmap=True, mode='denywrite'):
        from astropy.io import fits
        with fits.open(file, memmap=memmap, mode=mode) as hdul:
            hdu = hdul[ext]
            data = hdu.data
            header = hdu.header
        array = da.from_array(data, chunks=self.chunks)
        self.data = array
        self.header = header

    def read_fitsio(self, file: str, ext=0):
        from fitsio import FITS
        with FITS(file) as hdul:
            hdu = hdul[ext]
            data = hdu.read()
            header = hdu.read_header()
        array = da.from_array(data, chunks=self.chunks)
        self.data = array
        self.header = header

    def write_fits(self, file, header=None, verbose=True, **kwargs):
        """Write Dask Array to FITS file.

        Args:
            file (str): Name of FITS file.
            header (header, optional): FITS header. Defaults to None.
            **kwargs:
        """
        from astropy.io import fits
        if header is None:
            header = self.header
        tmp_file, z_data = self.write_tmp_zarr(file, self.data, self.verbose)
        hdu = fits.PrimaryHDU(z_data, header=header)
        hdu.writeto(file, **kwargs)
        if self.verbose:
            print(f'Wrote FITS file: {file}')
        shutil.rmtree(tmp_file)

    def write_fitsio(self, file, header=None, verbose=True, **kwargs):
        """Write Dask Array to FITS file.

        Args:
            file (str): Name of FITS file.
            header (header, optional): FITS header. Defaults to None.
            **kwargs:
        """
        from fitsio import FITS
        if header is None:
            header = self.header
        tmp_file, z_data = self.write_tmp_zarr(file, self.data, self.verbose)
        hdu = FITS(file, mode='rw')
        hdu.write(z_data, header=header, **kwargs)
        hdu.close()
        if self.verbose:
            print(f'Wrote FITS file: {file}')
        shutil.rmtree(tmp_file)

    @staticmethod
    def write_tmp_zarr(file, data, verbose=False):
        import zarr
        tmp_file = file.replace('.fits', '_tmp.zarr')
        if verbose:
            print(f'Writing temporary zarr file: {tmp_file}')
        data.to_zarr(tmp_file)
        z_data = zarr.open(tmp_file, mode='r')
        return tmp_file, z_data