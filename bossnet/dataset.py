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
    - LAMOST: represents the data source as LAMOST (Large Sky Area Multi-Object Fiber Spectroscopic Telescope).
    """
    BOSS="boss"
    LAMOSTDR7="lamost_dr7"
    LAMOSTDR8="lamost_dr8"

def extract_healpix_paths(hdul: fits.HDUList) -> List[str]:
    """
    A function that extracts a list of HEALPix paths from an astropy `fits.HDUList` object.

    Args:
    - hdul: fits.HDUList, an astropy `fits.HDUList` object containing the HEALPix paths.

    Returns:
    - paths: List[str], a list of HEALPix paths extracted from the input `fits.HDUList` object.
    """
    data = hdul[1].data
    paths = data["HEALPIX_PATH"].tolist()
    return paths

def join_healpix_paths(root_path: str, healpix_paths: str) -> List[str]:
    """
    A function that joins a root path and a list of HEALPix paths to create a list of file paths.

    Args:
    - root_path: str, the root path to be joined with the HEALPix paths.
    - healpix_paths: List[str], a list of HEALPix paths to be joined with the root path.

    Returns:
    - paths: List[str], a list of file paths created by joining the input root path with the input HEALPix paths.
    """
    return [os.path.join(root_path, healpix_path.replace("$MWM_HEALPIX/", "")) for healpix_path in healpix_paths]

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

def get_fits_function(data_source: DataSource) -> Callable:
    """ 
    The function get_fits_function returns the appropriate function for opening a FITS file based on the data source specified.

    Args:
    - data_source: Enum, The data source enum specifying which type of FITS file to open.

    Returns:
    - A Callable object representing the function to open the FITS file specified by data_source.
    """
    if data_source == DataSource.BOSS:
        return open_boss_fits
    if data_source == DataSource.LAMOSTDR7:
        return open_lamost_fitsDR7
    if data_source == DataSource.LAMOSTDR8:
        return open_lamost_fitsDR8


class BOSSDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        fits_table_path: str, 
        data_path: str, 
        data_source: Optional[Union[DataSource, str]]=DataSource.BOSS,
    ) -> None:
        """
        Initializes a BOSSDataset instance with paths to the FITS table and the directory containing the 
        LAMOST or BOSS FITS files, as well as optional metadata statistics, data source, and noise flux flag.
        
        Args:
        - fits_table_path: str, The path to the FITS table containing metadata for the LAMOST or BOSS FITS files.
        - data_path: str, The path to the directory containing the LAMOST or BOSS FITS files.
        - mstats: Optional[MetadataStats], Optional metadata statistics to normalize the metadata.
        - data_source: Optional[Union[DataSource, str]], Optional data source to specify which function to use 
          for opening the FITS files.
        - noise_flux: Optional[bool], Optional flag to add noise to the flux values.

        Returns:
        None
        """
        with fits.open(fits_table_path) as hdul:
            healpix_paths = extract_healpix_paths(hdul)

        # TODO: Write way for Lamost to be used.
        self.flux_paths = join_healpix_paths(data_path, healpix_paths)
        self.open_fits = get_fits_function(data_source)

    
    def __getitem__(self, index):
        """
        Returns the flux and metadata for a specified index in the dataset.

        Args:
        - index: int, The index of the LAMOST or BOSS FITS file in the dataset.

        Returns:
        - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.
        - metadata: torch.Tensor, A torch.Tensor representing the metadata values of the data.
        """
        fits_file_path = self.flux_paths[index]
        flux, error, wavlen = self.open_fits(fits_file_path)
        flux = flux[None, :]
        return flux, error, wavlen
  
    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        - length: int, The number of LAMOST or BOSS FITS files in the dataset.
        """
        return len(self.flux_paths)

# TODO: Write appogee dataset class.
# TODO: Write get_dataset function.
class APOGEEDataset(torch.utils.data.Dataset):
    def __init__(self):
        ...
    
    def __getitem__(self, i):
        ...
    
    def __len__(self):
        ...