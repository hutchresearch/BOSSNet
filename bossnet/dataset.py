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
from utils import DataStats

_MIN_WL = 3800
_MAX_WL = 8900
_FLUX_LEN = 3900

@dataclass(frozen=True)
class MetadataStats:
    """
    A dataclass that holds metadata statistics related to a all the metadata values.

    Attributes:
    - PARALLAX: DataStats, metadata stats for parallax measurements.
    - G: DataStats, metadata stats for G band measurements.
    - BP: DataStats, metadata stats for BP band measurements.
    - RP: DataStats, metadata stats for RP band measurements.
    - J: DataStats, metadata stats for J band measurements.
    - H: DataStats, metadata stats for H band measurements.
    - K: DataStats, metadata stats for K band measurements.
    """
    PARALLAX: DataStats
    G: DataStats
    BP: DataStats
    RP: DataStats
    J: DataStats
    H: DataStats
    K: DataStats

class DataSource(str, Enum):
    """
    An enumeration that represents the data source of a dataset.

    Enumeration members:
    - BOSS: represents the data source as BOSS (Baryon Oscillation Spectroscopic Survey).
    - LAMOST: represents the data source as LAMOST (Large Sky Area Multi-Object Fiber Spectroscopic Telescope).
    """
    BOSS="boss"
    LAMOST="lamost"

# Metadata values
mstats = MetadataStats(
    PARALLAX=DataStats(
        MEAN=1.9,
        STD=3.4,
        PNMAX=45.69664705882353,
        PNMIN=-3.328794117647059,
    ),
    G=DataStats(
        MEAN=13.5,
        STD=1.6,
        PNMAX=4.623286666666666,
        PNMIN=-4.688693125,
    ),
    BP=DataStats(
        MEAN=13.9,
        STD=1.8,
        PNMAX=4.623286666666666,
        PNMIN=-4.401857777777778,
    ),
    RP=DataStats(
        MEAN=12.9,
        STD=1.6,
        PNMAX=4.627214375,
        PNMIN=-4.56371375,
    ),
    J=DataStats(
        MEAN=12.1,
        STD=1.5,
        PNMAX=3.9579999999999997,
        PNMIN=-4.717333333333333,
    ),
    H=DataStats(
        MEAN=11.7,
        STD=1.5,
        PNMAX=4.306,
        PNMIN=-4.914666666666666,
    ),
    K=DataStats(
        MEAN=11.6,
        STD=1.5,
        PNMAX=3.9760000000000004,
        PNMIN=-4.948666666666667,
    )
)

def extract_metadata(hdul: fits.HDUList) -> torch.Tensor:
    """
    A function that extracts metadata from an astropy `fits.HDUList` object and returns it as a tensor.

    Args:
    - hdul: fits.HDUList, an astropy `fits.HDUList` object containing the metadata.

    Returns:
    - metadata: torch.Tensor, a tensor containing the extracted metadata. The tensor has shape `(n_samples, 7)`,
    where `n_samples` is the number of samples in the dataset, and the columns represent the following metadata
    statistics (in order): J band magnitude, H band magnitude, K band magnitude, parallax, G band magnitude,
    BP band magnitude, and RP band magnitude.
    """
    data = hdul[1].data
    two_mass = data["TWOMASS_MAG"]
    J = two_mass[:, 0]
    H = two_mass[:, 0]
    K = two_mass[:, 0]
    PARALLAX = data['PARALLAX']
    G = data['GAIA_G']
    BP = data['GAIA_BP']
    RP = data['GAIA_RP']
    
    metadata = np.column_stack([J, H, K, PARALLAX, G, BP, RP])
    metadata = torch.from_numpy(metadata)
    return metadata

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
    return [os.path.join(root_path, healpix_path) for healpix_path in healpix_paths]

def normalize_data(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    A function that normalizes input data using the provided mean and standard deviation.

    Args:
    - data: torch.Tensor, a tensor containing the data to be normalized.
    - mean: float, the mean value used for normalization.
    - std: float, the standard deviation used for normalization.

    Returns:
    - normalized_data: torch.Tensor, a tensor containing the normalized data.
    """
    data = (data - mean) / std
    return data

def normalize_col(col: torch.Tensor, data_stats: DataStats) -> torch.Tensor:
    """
    A function that normalizes a column of input data using the mean and standard deviation from a `DataStats` object.

    Args:
    - col: torch.Tensor, a tensor containing the column of data to be normalized.
    - data_stats: DataStats, a dataclass object containing the mean and standard deviation for the input data.

    Returns:
    - normalized_col: torch.Tensor, a tensor containing the normalized data.
    """
    col = col.float()
    normalized_data = normalize_data(col, data_stats.MEAN, data_stats.STD)
    return normalized_data

def normalize_metdata(metadata: torch.Tensor, mstats: MetadataStats) -> torch.Tensor:
    """
    A function that normalizes metadata using the mean and standard deviation from a `MetadataStats` object.

    Args:
    - metadata: torch.Tensor, a tensor containing the metadata to be normalized.
    - mstats: MetadataStats, a dataclass object containing the mean and standard deviation for each metadata column.

    Returns:
    - metadata: torch.Tensor, a tensor containing the normalized metadata.
    """
    metadata[:, 0] = normalize_col(metadata[:, 0], mstats.J)
    metadata[:, 1] = normalize_col(metadata[:, 1], mstats.H)
    metadata[:, 2] = normalize_col(metadata[:, 2], mstats.K)
    metadata[:, 3] = normalize_col(metadata[:, 3], mstats.PARALLAX)
    metadata[:, 4] = normalize_col(metadata[:, 4], mstats.G)
    metadata[:, 5] = normalize_col(metadata[:, 5], mstats.BP)
    metadata[:, 6] = normalize_col(metadata[:, 6], mstats.RP)
    return metadata

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

def open_lamost_fits(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The function open_lamost_fits opens a LAMOST FITS file and returns three torch.Tensors 
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
    if data_source == DataSource.LAMOST:
        return open_lamost_fits
    
def interpolate_flux(
    flux: torch.Tensor, wavelen: torch.Tensor, linear_grid: torch.Tensor
) -> torch.Tensor:
    """
    The function interpolate_flux takes in the flux, wavelength, and linear grid of a spectrum,
    interpolates the flux onto a new linear wavelength grid, and returns the interpolated flux as
    a torch.Tensor.

    Args:
    - flux: torch.Tensor, A torch.Tensor representing the flux values of the spectrum.
    - wavelen: torch.Tensor, A torch.Tensor representing the wavelength values of the spectrum.
    - linear_grid: torch.Tensor, A torch.Tensor representing the new linear wavelength grid to
      interpolate the flux onto.

    Returns:
    - interpolated_flux: torch.Tensor, A torch.Tensor representing the interpolated flux values
      of the spectrum on the new linear wavelength grid.
    """
    wavelen = wavelen[~torch.isnan(flux)]
    flux = flux[~torch.isnan(flux)]
    interpolated_flux = np.interp(linear_grid, wavelen, flux)
    interpolated_flux = torch.from_numpy(interpolated_flux)
    return interpolated_flux

def log_scale_flux(flux: torch.Tensor) -> torch.Tensor:
    """
    The function log_scale_flux applies a logarithmic scaling to the input flux tensor and clips the values
    to remove outliers.

    Args:
    - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.

    Returns:
    - flux: torch.Tensor, A torch.Tensor representing the logarithmically scaled flux values of the data.
    The values are clipped at the 95th percentile plus one to remove outliers.
    """
    s = 0.000001
    flux = torch.clip(flux, min=s, max=None)
    flux = torch.log(flux)
    perc_95 = torch.quantile(flux, 0.95)
    flux = torch.clip(flux, a_min=None, a_max=perc_95 + 1)
    return flux

def noise_flux(flux: torch.Tensor, error: torch.Tensor):
    """ 
    The function noise_flux adds Gaussian noise to the given flux tensor using the provided error tensor 
    as the standard deviation.

    Args:
    - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.
    - error: torch.Tensor, A torch.Tensor representing the error values of the data.

    Returns:
    - torch.Tensor, A torch.Tensor representing the noisy flux values of the data.
    """
    return flux + torch.randn(*flux.shape) * error

def fill_metadata_nans(metadata: torch.Tensor, default_values: torch.Tensor) -> torch.Tensor:
    """ 
    The `fill_metadata_nans` function fills NaN values in metadata with default values and returns the updated metadata tensor.
    This function also concatinates a mask to the send if the metdata.

    Args:
    - metadata: torch.Tensor, a tensor containing metadata values, some of which may be NaN.
    - default_values: torch.Tensor, a tensor containing default values to be used for any NaN values in the metadata tensor.

    Returns:
    - metadata: torch.Tensor, a tensor containing the updated metadata values, with any NaN values replaced by the corresponding values in `default_values`.
    """
    metadata_nans = torch.isnan(metadata)
    metadata = torch.nan_to_num(metadata)
    metadata = metadata + torch.where(
        ~metadata_nans, torch.Tensor([0.0]).type(torch.float32), default_values
    )
    metadata = torch.cat((metadata, (~metadata_nans).long()))
    return metadata

class BOSSDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        fits_table_path: str, 
        data_path: str, 
        mstats: Optional[MetadataStats]=mstats,
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
        
        self.noise = False
        
        with fits.open(fits_table_path) as hdul:
            metadata = extract_metadata(hdul)
            healpix_paths = extract_healpix_paths(hdul)

        self.flux_paths = join_healpix_paths(data_path, healpix_paths)
        self.metadata = normalize_metdata(metadata, mstats)
        self.metadata_defaults = torch.Tensor([
            mstats.J.PNMAX, mstats.H.PNMAX, mstats.K.PNMAX, mstats.PARALLAX.PNMIN,
            mstats.G.PNMAX, mstats.BP.PNMAX, mstats.RP.PNMAX
        ])
        self.linear_grid = torch.linspace(_MIN_WL, _MAX_WL, steps=_FLUX_LEN)
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
        
        flux = flux if not self.noise else noise_flux(flux, error)
        flux = interpolate_flux(flux, wavlen, self.linear_grid)
        flux = log_scale_flux(flux)
        flux = torch.reshape(flux, [1, -1])

        metadata = self.metadata[index]
        metadata = fill_metadata_nans(metadata, self.metadata_defaults)

        return flux, metadata
  
    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        - length: int, The number of LAMOST or BOSS FITS files in the dataset.
        """
        return len(self.flux_paths)

