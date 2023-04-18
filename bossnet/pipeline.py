import os
import io
import sys
import torch
from tqdm import tqdm
from bossnet.dataset import BOSSDataset
from bossnet.model import BossNet
from typing import Optional, Tuple
from bossnet.utils import DataStats, open_yaml, stats_from_dict
from dataclasses import dataclass
import tempfile
from collections import OrderedDict, namedtuple
import numpy as np
from functools import partial

@dataclass(frozen=True)
class StellarParameters:
    """
    The StellarParameters class is a dataclass that represents the statistical properties of 
    three stellar parameters: effective temperature (LOGTEFF), surface gravity (LOGG), and 
    metallicity (FEH). The class contains three attributes, each of which is an instance of 
    the DataStats class, representing the mean, standard deviation, post-normalization minimum, 
    and post-normalization maximum values of each parameter.

    Attributes:
    - LOGTEFF: DataStats, representing the statistical properties of the effective temperature.
    - LOGG: DataStats, representing the statistical properties of the surface gravity.
    - FEH: DataStats, representing the statistical properties of the metallicity.
    """
    LOGTEFF: DataStats
    LOGG: DataStats
    FEH: DataStats

stellar_parameter_stats = StellarParameters(
    LOGTEFF=DataStats(
        MEAN=,
        STD=,
        PNMAX=,
        PNMIN=,
    ),
    LOGG=DataStats(
        MEAN=,
        STD=,
        PNMAX=,
        PNMIN=,
    ),
    FEH=DataStats(
        MEAN=,
        STD=,
        PNMAX=,
        PNMIN=,
    ),
)

PredictionOutput = namedtuple('PredictionOutput', ['log_G', 'log_Teff', 'FeH'])

UncertaintyOutput = namedtuple('UncertaintyOutput', [
    'log_G_median', 'log_Teff_median', 'log_Feh_median',
    'log_G_std', 'log_Teff_std', 'log_Feh_std'
])

def unnormalize(X: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    This function takes in a PyTorch tensor X, and two scalar values mean and std, and returns the unnormalized tensor.

    Args:
    - X: torch.Tensor, input tensor to be unnormalized.
    - mean: float, mean value used for normalization.
    - std: float, standard deviation used for normalization.

    Returns:
    - torch.Tensor: Unnormalized tensor with the same shape as the input tensor X.
    """
    return X * std + mean

def unnormalize_predictions(predictions: torch.Tensor) -> torch.Tensor:
    """
    The unnormalize_predictions function takes a tensor X of shape (batch_size, 3) and unnormalizes 
    each of its three columns using the mean and standard deviation of the corresponding DataStats 
    objects. Specifically, the first column corresponds to LOGG, the second to LOGTEFF, and the third to FEH.

    Args:
    - predictions: torch.Tensor, Input tensor of shape (batch_size, 3).
    - stellar_parameter_stats: StellarParameters, an object containing the mean and standard deviation of 
      the three columns of X.

    Returns:
    - torch.Tensor: Output tensor of shape (batch_size, 3) where each column has been unnormalized using 
      the mean and standard deviation stored in stellar_parameter_stats.
    """
    predictions[:, 0] = unnormalize(predictions[:, 0], stellar_parameter_stats.LOGG.MEAN, stellar_parameter_stats.LOGG.STD)
    predictions[:, 1] = unnormalize(predictions[:, 1], stellar_parameter_stats.LOGTEFF.MEAN, stellar_parameter_stats.LOGTEFF.STD)
    predictions[:, 2] = unnormalize(predictions[:, 2], stellar_parameter_stats.FEH.MEAN, stellar_parameter_stats.FEH.STD)

    return predictions

def franken_load(load_path: str, chunks: int) -> OrderedDict:
    """
    Loads a PyTorch model from multiple binary files that were previously split.

    Args:
        load_path: str, The directory where the model chunks are located.
        chunks: int, The number of model chunks to load.

    Returns:
        A ordered dictionary containing the PyTorch model state.
    """

    def load_member(load_path: str, file_out: io.BufferedReader, file_i: int) -> None:
        """
        Reads a single chunk of the model from disk and writes it to a buffer.

        Args:
            load_path: str, The directory where the model chunk files are located.
            file_out: io.BufferedReader, The buffer where the model chunks are written.
            file_i: int, The index of the model chunk to read.

        """
        load_name = os.path.join(load_path, f"model_chunk_{file_i}")
        with open(load_name, "rb") as file_in:
            file_out.write(file_in.read())

    with tempfile.TemporaryDirectory() as tempdir:
        # Create a temporary file to write the model chunks.
        model_path = os.path.join(tempdir, "model.pt")
        with open(model_path, "wb") as file_out:
            # Load each model chunk and write it to the buffer.
            for i in range(chunks):
                load_member(load_path, file_out, i)
        
        # Load the PyTorch model from the buffer.
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    return state_dict

def create_uncertainties_batch(flux: torch.Tensor, error: torch.Tensor, num_uncertainty_draws: int) -> torch.Tensor:
    """
    Creates a batch of flux tensors with added noise from the specified error tensors.

    Args:
    - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.
    - error: torch.Tensor, A torch.Tensor representing the error values of the data.
    - num_uncertainty_draws: int, The number of times to draw noise samples to create a batch of flux tensors.

    Returns:
    - flux_with_noise: torch.Tensor, A torch.Tensor representing the batch of flux tensors with added noise from the 
      specified error tensors.
    """
    normal_sample = torch.randn((num_uncertainty_draws, *error.shape[-2:]))
    return flux + error * normal_sample

def interpolate_flux(
    flux_batch: torch.Tensor, wavelen: torch.Tensor, linear_grid: torch.Tensor
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
    interpolated_flux = torch.zeros(*flux_batch.shape[:-1], len(linear_grid))
    for i, flux in enumerate(flux_batch):
        _wavelen = wavelen[~torch.isnan(flux)]
        _flux = flux[~torch.isnan(flux)]
        _flux = np.interp(linear_grid, _wavelen, _flux)
        _flux = torch.from_numpy(_flux)
        interpolated_flux[i] = _flux
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
    flux = torch.clip(flux, min=None, max=perc_95 + 1)
    return flux

class Pipeline():
    """
    A class for running predictions using the BOSS dataset and BossNet model.

    Args:
        fits_table_path: str, The path to the FITS table.
        data_path: str, The path to the BOSS spectra data.
        output_path: Optional[str], The output path for writing predictions. If None,
            predictions are written to sys.stdout. Default is None.
        num_uncertainty_realizations: Optional[int], The number of uncertainty realizations
            to calculate. If 0, standard predictions are made. Default is 0.
        verbose: bool, Whether to print progress bars during prediction. Default is True.
    """
    def __init__(
        self, 
        fits_table_path: str, 
        data_path: str, 
        output_path: Optional[str] = None,
        num_uncertainty_draws: Optional[int] = 0,
        verbose: bool = False
    ) -> None:
        """
        Initializes the Pipeline object.

        Args:
            fits_table_path: str, The path to the FITS table.
            data_path: str, The path to the BOSS spectra data.
            output_path: Optional[str], The output path for writing predictions. If None,
                predictions are written to sys.stdout. Default is None.
            verbose: bool, Whether to print progress bars during prediction. Default is True.
        """
        self.verbose = verbose
        self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset: BOSSDataset = BOSSDataset(fits_table_path, data_path)
        self.model: BOSSDataset = BossNet()
        self._load_model()
        self.output_file = open(output_path, "a+") if output_path else sys.stdout
        self.num_uncertainty_draws: int = num_uncertainty_draws
        self.calculate_uncertainties: bool = num_uncertainty_draws > 0
        
        MIN_WL, MAX_WL, FLUX_LEN = 3800, 8900, 3900
        linear_grid = torch.linspace(MIN_WL, MAX_WL, steps=FLUX_LEN)
        self.interpolate_flux = partial(interpolate_flux, linear_grid=linear_grid)
    
    def _load_model(self):
        """
        Loads the BossNet model from disk.
        """
        # TODO: IF apogee, laod appogee model
        # model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "model_parts.pt")
        model_path = "/uufs/chpc.utah.edu/common/home/u6031723/logan_files/deconstructed_model/"
        state_dict = franken_load(model_path, 10)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self) -> None:
        """
        Runs the pipeline to generate predictions.

        Args:
            num_uncertainty_realizations: Optional[int], The number of uncertainty realizations
                to calculate. If 0, standard predictions are made. Default is 0.
        """
        loader = torch.utils.data.DataLoader(
            self.dataset,  
            num_workers=6, 
            shuffle=False,
        )

        self._write_header()
        for preds, uncertainty_preds in self._make_prediction(loader):
            self._write_prediction(preds, uncertainty_preds)
        self.output_file.close()
        
    def _make_prediction(self, loader: torch.utils.data.DataLoader):
        """
        Runs the BossNet model on a DataLoader of spectra and metadata.

        Args:
            loader: torch.utils.data.DataLoader, A DataLoader containing spectra and metadata.

        Yields:
            torch.Tensor: The predicted logg, logteff, and feh for each spectrum.
        """
        loader = tqdm(loader) if self.verbose else loader
        for spectra, error, wavlen in tqdm(loader):
            spectra, error, wavlen = spectra.to(self.device), error.to(self.device), wavlen.to(self.device)
            interp_spectra = self.interpolate_flux(spectra, wavlen)
            normalized_spectra = log_scale_flux(interp_spectra).float()
            normalized_prediction = self.model(normalized_spectra)
            prediction = unnormalize_predictions(normalized_prediction)
            prediction = prediction.squeeze()

            log_G = prediction[0].item()
            log_Teff = prediction[1].item()
            FeH = prediction[2].item()

            preds = PredictionOutput(
                log_G=log_G,
                log_Teff=log_Teff,
                FeH=FeH,
            )

            uncertainty_preds = None
            if self.calculate_uncertainties:
                uncertainties_batch = create_uncertainties_batch(spectra, error, self.num_uncertainty_draws)
                interp_uncertainties_batch = self.interpolate_flux(uncertainties_batch, wavlen)
                normalized_uncertainties_batch = log_scale_flux(interp_uncertainties_batch).float()
                normalized_predictions_batch = self.model(normalized_uncertainties_batch)
                prediction = unnormalize_predictions(normalized_predictions_batch)
                median = torch.median(prediction, axis=0)[0]
                std = torch.std(prediction, axis=0)

                log_G_median = median[0].item()
                log_Teff_median = median[1].item()
                log_Feh_median = median[2].item()

                log_G_std = std[0].item()
                log_Teff_std = std[1].item()
                log_Feh_std = std[2].item()
            
                uncertainty_preds = UncertaintyOutput(
                    log_G_median=log_G_median,
                    log_Teff_median=log_Teff_median,
                    log_Feh_median=log_Feh_median,
                    log_G_std=log_G_std,
                    log_Teff_std=log_Teff_std,
                    log_Feh_std=log_Feh_std
                )

            yield preds, uncertainty_preds

    def _write_header(self) -> None:
        """Writes the header of the output file, depending on whether uncertainties
        are being calculated or not.
        """
        if self.calculate_uncertainties:
            self.output_file.write("logg,logTeff,FeH,logg_median,logTeff_median,FeH_median,logg_std,logTeff_std,FeH_std\n")
        else:
            self.output_file.write("logg,logTeff,FeH\n")
    
    def _write_prediction(
        self, preds: PredictionOutput, uncertainty_preds: UncertaintyOutput
    ) -> None:
        """Writes a single prediction to the output file, along with the standard deviation
        if uncertainties are being calculated.
    
        Args:
        - prediction: torch.Tensor, A tensor containing the predicted target values.
        - std: Optional[torch.Tensor], A tensor containing the standard deviation for each
            predicted target value. This argument is optional and defaults to None if
            uncertainties are not being calculated.
        """
        logg, logteff, feh = preds
        if self.calculate_uncertainties:
            logg_median, logteff_median, feh_median, logg_std, logteff_std, feh_std = uncertainty_preds
            self.output_file.write((f"{logg},{logteff},{feh},{logg_median},{logteff_median},{feh_median},{logg_std},{logteff_std},{feh_std}\n"))
        else:
            self.output_file.write(f"{logg},{logteff},{feh}\n")
