"""
The dataset used for the BOSS Net model.
This file includes the BOSSDataset object and any relevant resources.

MIT License
Copyright (c) 2023 hutchresearch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import torch
from typing import List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from astropy.io import fits
import numpy as np
from enum import Enum

class DataSource(str, Enum):
    """
    An enumeration that represents the data source of a dataset.

    Enumeration members:
    - BOSS: represents the data source as BOSS (Baryon Oscillation Spectroscopic Survey).
    - LAMOSTDR7: represents the data source as LAMOST (Large Sky Area Multi-Object Fiber Spectroscopic Telescope).
    - LAMOSTDR8: represents the data source as LAMOST (Large Sky Area Multi-Object Fiber Spectroscopic Telescope).
    """
    BOSS="boss"
    APOGEE="apogee"
    LAMOSTDR7="lamost_dr7"
    LAMOSTDR8="lamost_dr8"

def reverse_inverse_error(inverse_error: np.array, default_error: int) -> np.array:
    """
    A function that calculates error values from inverse errors.

    Args:
    - inverse_error: np.array, a numpy array containing inverse error values.
    - default_error: int, an integer to use for error values that cannot be calculated.

    Returns:
    - error: np.array, a numpy array containing calculated error values.

    The function calculates error values from inverse error values by taking the square root of the reciprocal of each
    value in the input `inverse_error` array. The resulting array is then processed to replace any infinite or NaN values
    with a default error value or a multiple of the median non-infinite value in the array. The resulting array is returned
    as a numpy array.
    """
    np.seterr(all="ignore")
    inverse_error = np.nan_to_num(inverse_error)
    error = np.divide(1, inverse_error) ** 0.5
    if np.isinf(error).all():
        error = np.ones(*error.shape) * default_error
        error = error.astype(inverse_error.dtype)
    median_error = np.nanmedian(error[error != np.inf])
    error = np.clip(error, a_min=None, a_max=5 * median_error)
    error = np.where(np.isnan(error), 5 * median_error, error)
    return error

def open_lamost_fitsDR7(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The function open_lamost_fits opens a LAMOST DR7 FITS file and returns three torch.Tensors 
    representing the flux, error, and wavelength of the data.

    Args:
    - file_path: str, The path to the LAMOST FITS file to be opened.

    Returns:
    - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.
    - error: torch.Tensor, A torch.Tensor representing the error values of the data.
    - wavlen: torch.Tensor, A torch.Tensor representing the wavelength values of the data.
    """
    with fits.open(file_path) as hdul:
        flux = hdul[0].data[0, :].astype(np.float32)
        inverse_error = hdul[0].data[1, :].astype(np.float32)
        error = reverse_inverse_error(inverse_error, np.median(flux) * 0.1)
        wavlen = hdul[0].data[2, :].astype(np.float32)

    flux = torch.from_numpy(flux).float()
    error = torch.from_numpy(error).float()
    wavlen = torch.from_numpy(wavlen).float()

    return flux, error, wavlen

def open_lamost_fitsDR8(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The function open_lamost_fits opens a LAMOST DR8 FITS file and returns three torch.Tensors 
    representing the flux, error, and wavelength of the data.

    Args:
    - file_path: str, The path to the LAMOST FITS file to be opened.

    Returns:
    - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.
    - error: torch.Tensor, A torch.Tensor representing the error values of the data.
    - wavlen: torch.Tensor, A torch.Tensor representing the wavelength values of the data.
    """
    with fits.open(file_path) as hdul:
        flux = hdul[1].data[0][0].astype(np.float32)
        inverse_error = hdul[1].data[0][1].astype(np.float32)
        error = reverse_inverse_error(inverse_error, np.median(flux) * 0.1)
        wavlen = hdul[1].data[0][2].astype(np.float32)

    return flux, error, wavlen

def open_boss_fits(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ 
    The function open_boss_fits opens a BOSS FITS file and returns three torch.Tensors
    representing the flux, error, and wavelength of the data.

    Args:
    - file_path: str, The path to the BOSS FITS file to be opened.

    Returns:
    - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.
    - error: torch.Tensor, A torch.Tensor representing the error values of the data.
    - wavlen: torch.Tensor, A torch.Tensor representing the wavelength values of the data.
    """
    with fits.open(file_path) as hdul:
        spec = hdul[1].data
        flux = spec["flux"].astype(np.float32)
        inverse_error = spec["ivar"].astype(np.float32)
        error = reverse_inverse_error(inverse_error, np.median(flux) * 0.1)
        wavlen = 10 ** spec["loglam"].astype(np.float32)

    flux = torch.from_numpy(flux).float()
    error = torch.from_numpy(error).float()
    wavlen = torch.from_numpy(wavlen).float()

    return flux, error, wavlen

def open_apogee_fits(file_path: str) -> Tuple[str, np.array, np.array, np.array]:
    with fits.open(file_path) as hdul:
        # We use the first index to specify that we are using the coadded spectrum
        flux = hdul[1].data.astype(np.float32)[0]
        error = hdul[2].data.astype(np.float32)[0]
        wavlen = np.zeros_like(flux)
    flux = torch.from_numpy(flux).float()
    error = torch.from_numpy(error).float()

    return flux, error, wavlen

def get_fits_function(data_source: DataSource) -> Callable:
    """ 
    The function get_fits_function returns the appropriate function for opening a FITS file based on 
    the data source specified.

    Args:
    - data_source: Enum, The data source enum specifying which type of FITS file to open.

    Returns:
    - A Callable object representing the function to open the FITS file specified by data_source.
    """
    if data_source == DataSource.BOSS:
        return open_boss_fits
    if data_source == DataSource.APOGEE:
        return open_apogee_fits
    if data_source == DataSource.LAMOSTDR7:
        return open_lamost_fitsDR7
    if data_source == DataSource.LAMOSTDR8:
        return open_lamost_fitsDR8

def read_file_paths(file_path):
    """
    The function read_file_paths reads a text file containing file paths separated by new lines and 
    returns a list of those file paths.

    Args:
    - file_path: str, The path to the text file containing the file paths separated by new lines.

    Returns:
    - A list of strings, each string representing a file path read from the input file.
    """
    with open(file_path, 'r') as file:
        file_paths = [line.strip() for line in file]
    return file_paths

class BOSSDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        spectra_paths: str,
        data_source: Optional[Union[DataSource, str]]=DataSource.BOSS,
    ) -> None:
        """
        Initializes a BOSSDataset instance with a path to a plain text file containing paths to spectra FITS files 
        and the data source, which specifies which function to use for opening the FITS files.

        Args:
        - spectra_paths: str, The path to a plain text file containing paths to spectra FITS files.
        - data_source: Optional[Union[DataSource, str]], Optional data source to specify which function to use 
          for opening the FITS files. Default is DataSource.BOSS.

        Returns:
        None
        """
        self.flux_paths = read_file_paths(spectra_paths)
        self.open_fits = get_fits_function(data_source)

    def __getitem__(self, index):
        """
        Returns the flux, error and wavelength for a specified index in the dataset.

        Args:
        - index: int, The index of the FITS file in the dataset.

        Returns:
        - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.
        - error: torch.Tensor, A torch.Tensor representing the error values of the data.
        - wavelength: torch.Tensor, A torch.Tensor representing the wavelength values of the data.
        """
        fits_file_path = self.flux_paths[index]
        flux, error, wavlen = self.open_fits(fits_file_path)
        flux = np.nan_to_num(flux, nan=0.0)
        flux = torch.from_numpy(flux)
        flux = flux[None, :]
        return flux, error, wavlen
  
    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        - length: int, The number of FITS files in the dataset.
        """
        return len(self.flux_paths)
