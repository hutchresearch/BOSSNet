# BOSSNet

BossNet is a powerful and efficient pipeline for predicting stellar parameters (effective temperature, surface gravity, metallicity, and radial velocity).

The pipeline is designed to be easy-to-use, providing both accurate parameter predictions and estimates of uncertainties associated with these predictions.

## Usage

To run the pipeline, you need to call the main function from the command line and provide the required arguments:

```bash
python bossnet path_to_spectra_files -d data_source -o output_file -u num_uncertainty_draws -v
```

For example:

```bash
python main.py 'spectra' -d 'boss' -o 'predictions.txt' -u 100 -v
```

Here's what each argument does:

- `path_to_spectra_files`: The path to your spectra files
- `-d`, `--data_source`: The source of data. Default is 'boss'. Options: ['boss', 'lamost_dr7', 'lamost_dr8']
- `-o`, `--output_file`: The path to the file where the predictions will be saved. If not provided, the output will be printed to the console
- `-u`, `--num_uncertainty_draws`: The number of realizations to sample from the predictive distribution to calculate uncertainties. Default is 0
- `-v`, `--verbose`: If provided, the progress messages will be printed to the console
