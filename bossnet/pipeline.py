import os
import io
import sys
import torch
from dataset import BOSSDataset
from model import BossNet
from typing import Optional
from utils import DataStats
from dataclasses import dataclass

@dataclass(frozen=True)
class StellarParameters:
    TEFF: DataStats
    G: DataStats
    FEH: DataStats

stellar_parameter_stats = StellarParameters(
    LOGTEFF=DataStats(
        MEAN=3.8,
        STD=0.1,
        PNMIN=-6.324908332532,
        PNMAX=12.000000000000002
    ),
    LOGG=DataStats(
        MEAN=3.9,
        STD=0.8,
        PNMIN=-4.954011619091034,
        PNMAX=7.40750026702881
    ),
    FEH=DataStats(
        MEAN=-0.4,
        STD=0.5,
        PNMIN=-7.4961997985839846,
        PNMAX=4.449684190750122
    ),
)

def unnormalize(X: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return X * std + mean

def unnormalize_predictions(X: torch.Tensor) -> torch.Tensor:
    X[:, 0] = unnormalize(X[:, 0], stellar_parameter_stats.LOGG.STD, stellar_parameter_stats.G.MEAN)
    X[:, 1] = unnormalize(X[:, 1], stellar_parameter_stats.LOGTEFF.STD, stellar_parameter_stats.TEFF.MEAN)
    X[:, 2] = unnormalize(X[:, 2], stellar_parameter_stats.FEH.STD, stellar_parameter_stats.FEH.MEAN)

    return X

class UncertaintiesSampler(torch.utils.data.Sampler):
    """This is a sampler that will create a batch for each data element."""
    def __init__(self, dataset: torch.utils.data.Dataset, num_uncertainty_realizations: int):
        self.dataset = dataset
        self.num_uncertainty_realizations = num_uncertainty_realizations
        self.new_inds = [[i] * num_uncertainty_realizations for i in list(range(len(dataset)))]
  
    def __iter__(self):
        return iter(self.new_inds)
  
    def __len__(self):
        return len(self.dataset) // self.num_uncertainty_realizations

class Pipeline():
    def __init__(
        self, 
        fits_table_path: str, 
        data_path: str, 
        output_path: Optional[str] = None, 
        num_uncertainty_realizations: Optional[int] = 0
    ) -> None:
        self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset: BOSSDataset = BOSSDataset(fits_table_path, data_path)

        self.model: BOSSDataset = BossNet()
        self._load_model()

        self.output_path: str = output_path
        self.uncertainty_realizations: int = num_uncertainty_realizations
        self.calculate_uncertainties: bool = num_uncertainty_realizations > 0
        self.dataset.noise = self.calculate_uncertainties
        
    
    def _load_model(self):
        model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], "model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self) -> None:
        """Creates predictions for spectra in the fits table, and writes them.
        This function does a lot of the work of the class, both feeding the
        predictions through the model and writing them.
        """
        uncertainties_sampler = UncertaintiesSampler(
            dataset=self.dataset, 
            num_uncertainty_realizations=self.num_uncertainty_realizations
        ) if self.calculate_uncertainties else None

        
        f = open(self.output_path, "a+") if self.output_path else sys.stdout
        self._write_header(f)
        with torch.set_grad_enabled(False):
            loader = torch.utils.data.DataLoader(
                self.dataset, 
                batch_size=None, 
                num_workers=6, 
                batch_sampler=uncertainties_sampler
            )
            for spectra, mdata in loader:
                spectra, mdata, error = spectra.to(self.device), mdata.to(self.device), error.to(self.device)
                normalized_prediction = self.model(spectra, mdata)
                prediction = unnormalize_predictions(normalized_prediction)

                if self.calculate_uncertainties:
                    median = torch.median(prediction, axis=0)
                    std = torch.std(prediction, axis=0)
                    self._write_prediction(f, median, std)
                else:
                    self._write_prediction(f, prediction)


    def _write_header(self, f: io.TextIOWrapper) -> None:
        if self.calculate_uncertainties:
            f.write("logg,logTeff,FeH,logg_std,logTeff_std,FeH_std\n")
        else:
            f.write("logg,logTeff,FeH\n")
    
    def _write_prediction(
        self, f: io.TextIOWrapper, prediction: torch.Tensor, std: Optional[torch.Tensor]=None) -> None:
        logg, logteff, feh = prediction
        if self._do_uncertainties:
            logg_std, logteff_std, feh_std = std
            f.write((f"{logg},{logteff},{feh},{logg_std},{feh_std}\n"))
        else:
            f.write(f"{logg},{logteff},{feh}\n")

if __name__ == "__main__":
    pipeline = Pipeline("/research/hutchinson/workspace/sizemol/ml_astro_boss/data/sample_spAll/spAll-sample.fits", "./", "./")

