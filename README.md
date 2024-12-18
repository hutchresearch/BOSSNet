# BOSSNet
[![DOI](https://zenodo.org/badge/610963244.svg)](https://zenodo.org/doi/10.5281/zenodo.10453134)

*Credit: Logan Sizemore, Diego Llanes, Indie Cowan, Dyaln Huson*

**BOSSNet** is a pipeline for predicting stellar parameters, including effective temperature, surface gravity, metallicity, and radial velocity. It supports various astronomical surveys and allows users to compute predictions with optional uncertainty estimates.

[BOSS Net Paper](https://iopscience.iop.org/article/10.3847/1538-3881/ad291d/meta)

## Supported Surveys

BOSSNet supports the following data sources:

- **BOSS** (Baryon Oscillation Spectroscopic Survey)
- **APOGEE** (Apache Point Observatory Galactic Evolution Experiment)
- **LAMOST** (Large Sky Area Multi-Object Fiber Spectroscopic Telescope), including:
  - LAMOST DR7
  - LAMOST DR8
- **Gaia** (Global astrometric and spectroscopic survey), including:
  - Gaia XP (Extended Photometric Survey)
  - Gaia RVS (Radial Velocity Spectrometer)

## Input File Requirements

- **BOSS, APOGEE, and LAMOST (DR7/DR8)**: Provide a plain text file containing a list of paths to the FITS files.
- **Gaia**: Provide an ECSV file in the format found at:
  - [Gaia XP Continuous Mean Spectrum](http://cdn.gea.esac.esa.int/Gaia/gdr3/Spectroscopy/xp_continuous_mean_spectrum/)
  - [Gaia RVS Mean Spectrum](http://cdn.gea.esac.esa.int/Gaia/gdr3/Spectroscopy/rvs_mean_spectrum/)

## Usage

To execute the pipeline, use the following command structure:

```bash
python bossnet <spectra_paths> -d <data_source> -o <output_file> -u <num_uncertainty_draws> -b <batch_size> -w <num_workers> -v
```

### Positional Arguments

- **`spectra_paths`**: Path to a plain text file containing paths to the spectra used for predictions.

### Optional Arguments

- **`-h, --help`**: Display help information and exit.
- **`-d, --data_source`**: Source of data. Default is `boss`. Available options are:
  - `boss`
  - `apogee`
  - `lamost_dr7`
  - `lamost_dr8`
  - `gaia_xp`
  - `gaia_rvs`
- **`-o, --output_file`**: Path to the file where predictions will be saved. If not provided, the predictions will go to standard out.
- **`-u, --num_uncertainty_draws`**: Number of realizations to sample from the predictive distribution for uncertainty calculation. Default is `0`.
- **`-b, --batch_size`**: Batch size for the data loader. Adjust this based on system memory availability.
- **`-w, --num_workers`**: Number of workers for the data loader to utilize for parallel data loading.
- **`-v, --verbose`**: Print progress messages to the console for better visibility of the execution process.

### Example Commands

1. Predict stellar parameters for BOSS data and save results to a file:

   ```bash
   python bossnet spectra_paths.txt -d boss -o predictions.txt -v
   ```

2. Include uncertainty estimation with 100 draws:

   ```bash
   python bossnet spectra_paths.txt -d apogee -o predictions_with_uncertainties.txt -u 100 -v
   ```

3. Use Gaia XP data with specific batch size and workers:

   ```bash
   python bossnet gaia_spectra.csv.gz -d gaia_xp -b 32 -w 4 -o gaia_predictions.txt
   ```

## Output

The output file (if specified) will contain predictions for each spectrum provided in the input file, including calculated stellar parameters and optional uncertainty estimates.