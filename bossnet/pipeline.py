"""
The pipeline used for the BOSS Net, APOGEE Net, and GAIA Net models.
This file includes the Pipeline object and any relevant resources.

MIT License
Copyright (c) 2024 hutchresearch

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
import numpy as np
from typing import Optional, List, Tuple
import tempfile
from collections import OrderedDict
from abc import ABC, abstractmethod
from dataset import BOSSDataset, GAIARVSDataset, GAIAXpDataset, boss_collate
from model.boss_net import BossNet
from utils import open_yaml, Stats
from tqdm import tqdm


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

def log_scale_flux_batch(flux_bath: torch.Tensor) -> torch.Tensor:
    """log_scale_flux applies a logarithmic scaling to the input flux tensor and clips the values
    to remove outliers for each element in a batch.

    Args:
    - flux_batch: torch.Tensor, A torch.Tensor representing a batch of flux data.

    Returns:
    - log_flux_batch: torch.Tensor, A torch.Tensor representing the logarithmically scaled flux values of the data
      in the batch. The values are clipped at the 95th percentile plus one to remove outliers.
    """
    log_flux_batch = torch.zeros_like(flux_bath)
    for i, flux in enumerate(flux_bath):
        log_flux_batch[i] = log_scale_flux(flux)
    return log_flux_batch

def interpolate_flux(
    flux_batch: torch.Tensor, wavelen: torch.Tensor,linear_grid: torch.Tensor) -> torch.Tensor:
    """
    The function interpolate_flux takes in the flux, wavelength, and linear grid of a spectrum,
    interpolates the flux onto a new linear wavelength grid, and returns the interpolated flux as
    a torch.Tensor.

    Args:
    - flux_batch: torch.Tensor, A torch.Tensor representing the flux values of the spectrum.
    - wavelen: torch.Tensor, A torch.Tensor representing the wavelength values of the spectrum.
    - linear_grid: torch.Tensor, A torch.Tensor representing the new linear wavelength grid to
      interpolate the flux onto.

    Returns:
    - interpolated_flux: torch.Tensor, A torch.Tensor representing the interpolated flux values
      of the spectrum on the new linear wavelength grid.
    """
    
    interpolated_flux = torch.zeros(*flux_batch.shape[:-1], len(linear_grid))
    for i, flux in enumerate(flux_batch):
        _wavelen = wavelen[i][~torch.isnan(flux)]
        _flux = flux[~torch.isnan(flux)]
        _flux = np.interp(linear_grid, _wavelen, _flux)
        _flux = torch.from_numpy(_flux)
        interpolated_flux[i] = _flux
    return interpolated_flux

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

def unnormalize_predictions(predictions: torch.Tensor, param_names: List[str], stats: Stats) -> torch.Tensor:
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
    for i, param_name in enumerate(param_names):
        param_stat = getattr(stats, param_name)
        predictions[:, i] = unnormalize(predictions[:, i], param_stat.MEAN, param_stat.STD)
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

def create_uncertainties_batch(flux_batch: torch.Tensor, error_batch: torch.Tensor) -> torch.Tensor:
    """
    Creates a batch of flux tensors with added noise from the specified error tensors.

    Args:
    - flux: torch.Tensor, A torch.Tensor representing the flux values of the data.
    - error: torch.Tensor, A torch.Tensor representing the error values of the data.

    Returns:
    - flux_with_noise: torch.Tensor, A torch.Tensor representing the batch of flux tensors with added noise from the 
      specified error tensors.
    """
    normal_sample = torch.randn_like(flux_batch)
    normal_sample[flux_batch.isnan()] = torch.nan
    return flux_batch + error_batch * normal_sample


class Pipeline(ABC):
    def __init__(
        self, 
        deconstructed_model_dir: str,
        dataset: torch.utils.data.Dataset,
        param_list: List[str],
        stats: Stats,
        output_file: Optional[str] = None,
        num_uncertainty_draws: Optional[int] = 0,
        collate_fn: Optional[callable] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the Pipeline object with the specified parameters, including loading the model and 
        setting up the dataset, output file, and other settings.

        Args:
        - deconstructed_model_dir: str, Directory path for the deconstructed model.
        - dataset: torch.utils.data.Dataset), The dataset used for predictions.
        - param_list: List[str], List of parameter names to include in predictions.
        - stats: Stats, Statistical information for unnormalizing the data.
        - output_file: Optional[str], Path to the output file for predictions (defaults to standard output).
        - num_uncertainty_draws: Optional[int], Number of draws for uncertainty calculations (default is 0, no uncertainty).
        - collate_fn: Optional[callable], Optional function to collate data batches for DataLoader.
        - verbose: bool, If True, prints progress information (default is False).
        """

        self.verbose = verbose
        self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset = dataset
        self.model = self._load_model(deconstructed_model_dir)
        
        self.param_list = param_list
        self.stats = stats

        # If no output file specified, then predictions will be written to standard out.
        self.output_file = open(output_file, "a+") if output_file else sys.stdout

        self.num_uncertainty_draws: int = num_uncertainty_draws
        self.calculate_uncertainties: bool = num_uncertainty_draws > 0

        self.collate_fn = collate_fn

    def _load_model(self, deconstructed_model_dir: str) -> BossNet:
        """
        Loads the BossNet model from disk, including its configuration and trained weights.

        Args:
        - deconstructed_model_dir: str, The directory containing the model's configuration and weights.

        Returns:
        - model: BossNet, The loaded model.
        """
        model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], deconstructed_model_dir)
        model_args = open_yaml(os.path.join(model_path, "model_args.yaml"))
        model = BossNet(**model_args)
        state_dict = franken_load(model_path, 10)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, batch_size: int, num_workers: int) -> None:
        """
        Runs the pipeline to generate predictions on the provided dataset. Optionally calculates uncertainties 
        for each prediction if the `num_uncertainty_draws` is greater than zero.

        This method iterates over the dataset in batches, makes predictions, and writes the results to the output file.

        Returns:
        - None
        """
        loader = torch.utils.data.DataLoader(
            self.dataset,  
            num_workers=num_workers, 
            shuffle=False,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
        )

        self._write_header()
        loader = tqdm(loader, desc="Predicting: ") if self.verbose else loader
        for source_id_batch, flux_batch, error_batch, wavelen_batch in loader:
            prediction_batch = self._make_prediction(flux_batch, wavelen_batch)
            if self.calculate_uncertainties:

                if (error_batch == -1).all():
                    raise ValueError("Cannot calculate uncertainties: no error for specified data source.")
                
                prediction_batch_median, prediction_batch_std = self._make_uncertainty_prediction(flux_batch, error_batch, wavelen_batch)
                self._write_prediction(source_id_batch, prediction_batch, prediction_batch_median, prediction_batch_std)
            else:
                self._write_prediction(source_id_batch, prediction_batch)
        self.output_file.close()

    def _write_header(self) -> None:
        """
        Writes the header of the output file, depending on whether uncertainties
        are being calculated or not. The header includes the parameter names and, if applicable, 
        their median and standard deviation.

        Returns:
        - None
        """
        if self.calculate_uncertainties:
            header = ",".join(["id"] + self.param_list) + ","
            header += ",".join([s + "_median" for s in self.param_list]) + ","
            header += ",".join([s + "_std" for s in self.param_list]) + "\n"
        else:
            header = ",".join(["id"] + self.param_list) + "\n"
        self.output_file.write(header)
    
    def _write_prediction(
        self, 
        source_id_batch: torch.Tensor,
        prediction_batch: torch.Tensor, 
        prediction_batch_median: Optional[torch.Tensor]=None, 
        prediction_batch_std: Optional[torch.Tensor]=None,
    ) -> None:
        """
        Writes the predictions (and optionally their uncertainties) to the output file in CSV format.

        Args:
        - source_id_batch: torch.Tensor, The batch of source IDs.
        - prediction_batch: torch.Tensor, The batch of predicted values.
        - prediction_batch_median: Optional[torch.Tensor], The batch of median uncertainty predictions (default is None).
        - prediction_batch_std: Optional[torch.Tensor], The batch of standard deviation uncertainty predictions (default is None).

        Returns:
        - None
        """
        if isinstance(source_id_batch, torch.Tensor):
            source_id_batch = source_id_batch.tolist()

        for i in range(len(source_id_batch)):
            prediction_as_str = [str(pred) for pred in prediction_batch[i].cpu().tolist()]
            csv_row = ",".join([str(source_id_batch[i])] + prediction_as_str)
            if prediction_batch_median is not None:
                median_as_str = [str(pred) for pred in prediction_batch_median[i].cpu().tolist()]
                csv_row += "," + ",".join(median_as_str)

                std_as_str = [str(pred) for pred in prediction_batch_std[i].cpu().tolist()]
                csv_row += "," + ",".join(std_as_str)
            self.output_file.write(csv_row + "\n")
        
    def _make_uncertainty_prediction(
        self, 
        flux_batch: torch.Tensor, 
        error_batch: torch.Tensor, 
        wavelen_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates predictions by sampling multiple uncertainty batches and calculates the median and 
        standard deviation of the predictions.

        Args:
        - flux_batch: torch.Tensor, A batch of flux values.
        - error_batch: torch.Tensor, A batch of error values associated with the fluxes.
        - wavelen_batch: torch.Tensor, A batch of wavelength values.

        Returns:
        - prediction_batch_median: torch.Tensor, The median of the predictions across uncertainty draws.
        - prediction_batch_std: torch.Tensor, The standard deviation of the predictions across uncertainty draws.
        """
        predictions = []

        for _ in range(self.num_uncertainty_draws):
            uncertainties_batch = create_uncertainties_batch(flux_batch, error_batch)
           
            prediction_batch = self._make_prediction(uncertainties_batch, wavelen_batch)
            predictions.append(prediction_batch.detach().cpu())
        
        predictions = torch.stack(predictions)
        prediction_batch_median = torch.median(predictions, dim=0)[0]
        prediction_batch_std = torch.std(predictions, dim=0)
        return prediction_batch_median, prediction_batch_std
    
    @abstractmethod
    def _make_prediction(self, flux_batch: torch.Tensor, wavelen_batch: torch.Tensor) -> torch.Tensor:
        """
        Responsible for generating predictions based on the provided flux and wavelength batches.

        Args:
        - flux_batch: torch.Tensor, A batch of flux values.
        - wavelen_batch: torch.Tensor, A batch of wavelength values.

        Returns:
        - torch.Tensor: A tensor containing the predictions for the given batch.
        """



class BOSSNETPipeline(Pipeline):
    def __init__(
        self,
        spectra_paths: str,
        data_source: str,
        **kwargs,
    ) -> None:
        deconstructed_model_dir = "model_boss"
        param_list = ["logg", "logTeff", "FeH", "rv"]
        dataset = BOSSDataset(file_path=spectra_paths,data_source=data_source)

        super().__init__(
            deconstructed_model_dir=deconstructed_model_dir, 
            dataset=dataset, 
            param_list=param_list, 
            collate_fn=boss_collate,
            **kwargs
        )

        MIN_WL, MAX_WL, FLUX_LEN = 3800, 8900, 3900
        self.linear_grid = torch.linspace(MIN_WL, MAX_WL, steps=FLUX_LEN)
    
    def _make_prediction(self, flux_batch: torch.Tensor, wavelen_batch: torch.Tensor) -> None:
        interp_spectra = interpolate_flux(flux_batch, wavelen_batch, self.linear_grid)
        normalized_spectra = log_scale_flux_batch(interp_spectra).float()

        normalized_prediction = self.model(normalized_spectra[:, None, :].to(self.device))
        prediction_batch = unnormalize_predictions(normalized_prediction, self.param_list, self.stats)
        return prediction_batch



class ApogeeNetPipeline(Pipeline):
    def __init__(
        self, spectra_paths: str, **kwargs) -> None:
        deconstructed_model_dir = "model_apogee"
        param_list = ["logg", "logTeff", "FeH"]
        dataset = BOSSDataset(file_path=spectra_paths, data_source="apogee")

        super().__init__(
            deconstructed_model_dir=deconstructed_model_dir, 
            dataset=dataset, 
            param_list=param_list,
            **kwargs,
        )
    
    def _make_prediction(self, flux_batch: torch.Tensor, wavelen_batch: torch.Tensor) -> torch.Tensor:
        normalized_spectra = log_scale_flux_batch(flux_batch).float()

        normalized_prediction = self.model(normalized_spectra[:, None, :].to(self.device))
        prediction_batch = unnormalize_predictions(normalized_prediction, self.param_list, self.stats)
        return prediction_batch


class GAIAPipeline(Pipeline):
    def __init__(self, **kwargs) -> None:
        super().__init__(param_list=["logg", "logTeff", "FeH"], **kwargs)
    
    def _make_prediction(self, flux_batch: torch.Tensor, wavelen_batch) -> torch.Tensor:
        normalized_prediction = self.model(flux_batch[:, None, :].to(self.device))
        prediction_batch = unnormalize_predictions(normalized_prediction,  self.param_list, self.stats)
        return prediction_batch


class GAIARVSPipeline(GAIAPipeline):
    def __init__(self, file_path: str, **kwargs) -> None:
        dataset = GAIARVSDataset(file_path=file_path)
        super().__init__(deconstructed_model_dir="model_gaia_rvs", dataset=dataset, **kwargs)


class GAIAXpPipeline(GAIAPipeline):
    def __init__(self, file_path, **kwargs) -> None:
        dataset = GAIAXpDataset(file_path=file_path)
        super().__init__(deconstructed_model_dir="model_gaia_xp", dataset=dataset, **kwargs)
