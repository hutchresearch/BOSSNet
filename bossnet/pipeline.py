"""
The pipeline used for the BOSS Net model.
This file includes the Pipeline object and any relevant resources.

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
import io
import sys
import torch
from tqdm import tqdm
from bossnet.dataset import BOSSDataset
from bossnet.model import BossNet
from typing import Optional
from dataclasses import dataclass
import tempfile
from collections import OrderedDict, namedtuple
import numpy as np
from functools import partial

@dataclass(frozen=True)
class DataStats:
    """
    A dataclass that holds stellar parameter statistics related to a single value.

    Attributes:
    - MEAN: float, the mean value of the value.
    - STD: float, the standard deviation of the value.
    - PNMAX: float, the post-normalization maximum value of the value.
    - PNMIN: float, the post-normalization minimum value of the value.
    """
    MEAN: float
    STD: float
    PNMAX: float
    PNMIN: float

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
    - RV: DataStats, representing the statistical properties of the radial velocity.
    """
    LOGTEFF: DataStats
    LOGG: DataStats
    FEH: DataStats

# Unnormalization values for the stellar parameters.
stellar_parameter_stats = StellarParameters(
    FEH=DataStats(
        MEAN=-0.2,
        PNMAX=4.568167,
        PNMIN=-8.926531221275827,
        STD=0.3,
    ),
    LOGG=DataStats(
        MEAN=3.2,
        PNMAX=3.3389749999999996,
        PNMIN=-3.2758333384990697,
        STD=1.2,
    ),
    LOGTEFF=DataStats(
        MEAN=3.7,
        PNMAX=9.387230328989702,
        PNMIN=-5.2989908487604165,
        STD=0.1,
    ),
)
# Data structure for the output of the model.
PredictionOutput = namedtuple('PredictionOutput', ['log_G', 'log_Teff', 'FeH'])

# Data structure for uncertainty predictions.
UncertaintyOutput = namedtuple('UncertaintyOutput', [
    'log_G_median', 'log_Teff_median', 'Feh_median', 
    'log_G_std', 'log_Teff_std', 'Feh_std', 
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
    objects. Specifically, the first column corresponds to LOGG, the second to LOGTEFF, the third to FEH,
    and the fourth to RV.

    Args:
    - predictions: torch.Tensor, Input tensor of shape (batch_size, 4).
    - stellar_parameter_stats: StellarParameters, an object containing the mean and standard deviation of 
      the three columns of X.

    Returns:
    - torch.Tensor: Output tensor of shape (batch_size, 4) where each column has been unnormalized using 
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
    This class defines the pipeline for running predictions using the BossNet model.

    Args:
    - spectra_paths (str): The paths to the input spectra files.
    - data_source (str, optional): The source of the data, default is "boss".
    - output_file (str, optional): The file to which predictions will be written. If no output file is specified, 
      then predictions will be written to standard out.
    - num_uncertainty_draws (int, optional): The number of draws for uncertainty calculation, default is 0. If greater 
      than 0, then uncertainties will be calculated.
    - verbose (bool, optional): If set to True, verbose output will be enabled, default is False.

    Attributes:
    - verbose (bool): Verbose output flag.
    - device (torch.device): The device to which tensors will be sent.
    - dataset (BOSSDataset): The BOSS dataset.
    - model (torch.nn.Module): The BossNet model.
    - output_file (TextIO): The file to which predictions will be written.
    - num_uncertainty_draws (int): The number of draws for uncertainty calculation.
    - calculate_uncertainties (bool): A flag that indicates whether or not to calculate uncertainties.

    Methods:
    - _load_model(): Loads the BossNet model from disk.
    - predict(): Runs the pipeline to generate predictions.
    - _make_prediction(loader: torch.utils.data.DataLoader): Makes predictions for the given data loader.
    - _write_header(): Writes the header of the output file.
    - _write_prediction(file_path: str, preds: PredictionOutput, uncertainty_preds: UncertaintyOutput): Writes the predictions for a given input to the output file.
    """
    def __init__(
        self,
        spectra_paths: str,
        data_source: str = "apogee",
        output_file: Optional[str] = None,
        num_uncertainty_draws: Optional[int] = 0,
        verbose: bool = False
    ) -> None:

        self.verbose = verbose
        self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset: BOSSDataset = BOSSDataset(spectra_paths, data_source)
        self.model: torch.nn.module = BossNet()
        self._load_model()

        # If no output file specified, then predictions will be written to standard out.
        self.output_file = open(output_file, "a+") if output_file else sys.stdout

        self.num_uncertainty_draws: int = num_uncertainty_draws
        self.calculate_uncertainties: bool = num_uncertainty_draws > 0
        
    def _load_model(self):
        """
        Loads the BossNet model from disk.
        """
        model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "deconstructed_model")
        state_dict = franken_load(model_path, 10)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self) -> None:
        """
        Runs the pipeline to generate predictions.
        """
        loader = torch.utils.data.DataLoader(
            self.dataset,  
            num_workers=6, 
            shuffle=False,
        )

        prediction_generator = self._make_prediction(loader)

        self._write_header()
        for path in self.dataset.flux_paths:
            preds, uncertainty_preds = next(prediction_generator)
            if preds:
                self._write_prediction(path, preds, uncertainty_preds)
        self.output_file.close()
        
    def _make_prediction(self, loader: torch.utils.data.DataLoader):
        """
        Generates predictions and their uncertainties for a given data loader.

        Args:
            loader (torch.utils.data.DataLoader): A DataLoader object that loads the data for prediction.

        Yields:
            preds (PredictionOutput): The prediction output for each data input, including 
                the predicted logg, logTeff, FeH, and rv values.

            uncertainty_preds (UncertaintyOutput): The uncertainties associated with the predictions,
                including median values and standard deviations of the predicted logg, 
                logTeff, FeH, and rv values. This will only be yielded if self.calculate_uncertainties 
                is set to True, else None is yielded.

        Raises:
            OSError: If there is an issue loading the data (like missing or corrupt FITS file).

        """

        loader = iter(tqdm(loader, desc="Evaluating Model: ", file=sys.stderr) if self.verbose else loader)
        while loader:

            spectra, error, wavlen = next(loader)

            # log scale spectra
            normalized_spectra = log_scale_flux(spectra).float()

            # Calculate and unnormalize steller parameter predictions
            normalized_prediction = self.model(normalized_spectra.to(self.device))
            prediction = unnormalize_predictions(normalized_prediction)
            prediction = prediction.squeeze()

            # Unpack stellar parameters
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
                # Get batch of noised spectra
                uncertainties_batch = create_uncertainties_batch(spectra, error, self.num_uncertainty_draws)

                # scale sprectra
                normalized_uncertainties_batch = log_scale_flux(uncertainties_batch).float()

                # Calculate and unnormalize stellar parameters predictions
                normalized_predictions_batch = self.model(normalized_uncertainties_batch.to(self.device))
                prediction = unnormalize_predictions(normalized_predictions_batch)

                # Calculate the median and std for each stellar parameter
                median = torch.median(prediction, axis=0)[0]
                std = torch.std(prediction, axis=0)

                # Unpack medians
                log_G_median = median[0].item()
                log_Teff_median = median[1].item()
                Feh_median = median[2].item()

                # Unpack stds
                log_G_std = std[0].item()
                log_Teff_std = std[1].item()
                Feh_std = std[2].item()
            
                uncertainty_preds = UncertaintyOutput(
                    log_G_median=log_G_median,
                    log_Teff_median=log_Teff_median,
                    Feh_median=Feh_median,
                    log_G_std=log_G_std,
                    log_Teff_std=log_Teff_std,
                    Feh_std=Feh_std,
                )

            yield preds, uncertainty_preds

    def _write_header(self) -> None:
        """Writes the header of the output file, depending on whether uncertainties
        are being calculated or not.
        """
        if self.calculate_uncertainties:
            self.output_file.write((
                "path,logg,logTeff,FeH,rv,"
                "logg_median,logTeff_median,FeH_median,rv_median,"
                "logg_std,logTeff_std,FeH_std,rv_std\n"))
        else:
            self.output_file.write("path,logg,logTeff,FeH,rv\n")
    
    def _write_prediction(
        self, file_path: str, preds: PredictionOutput, uncertainty_preds: UncertaintyOutput
    ) -> None:
        """
        Writes the predictions for a given input to the output file.

        Args:
        - file_path (str): The path of the input file.
        - preds (Tuple[float, float, float, float]): A tuple of predicted values,
        containing the predicted logg, logteff, feh, and rv values, in that order.
        - uncertainty_preds (Optional[Tuple[float, float, float, float, float, float, float, float]]): A tuple of
        predicted uncertainties for the input file, containing the median values and standard
        deviations of the predicted logg, logteff, feh, and rv values, in that order. This argument is
        optional, and if it is not provided, the method will not output the uncertainty predictions
        to the output file.

        Returns:
        - None: This method has no return value.

        Output:
        - This method writes a line to the output file, with comma-separated values for the input file
        path, predicted logg, logteff, and feh values. If the `uncertainty_preds` argument is provided,
        this line will also include the median values and standard deviations of the predicted
        logg, logteff, feh, and rv values, in that order, separated by commas.
        """
        logg, logteff, feh = preds
        if self.calculate_uncertainties:
            logg_median, logteff_median, feh_median, logg_std, logteff_std, feh_std = uncertainty_preds
            self.output_file.write((f"{file_path},{logg},{logteff},{feh},{logg_median},{logteff_median},{feh_median},{logg_std},{logteff_std},{feh_std},\n"))
        else:
            self.output_file.write(f"{file_path},{logg},{logteff},{feh}\n")
