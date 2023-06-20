import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pipeline import Pipeline

def parse_args():
    """Parses command line arguments for running the model predictions."""
    parser = argparse.ArgumentParser(description="Run predictions on fits files and save to a file.")
    
    parser.add_argument("spectra_paths", type=str,
                        help="Path to plain text file containing paths to spectra used for predictions.")

    parser.add_argument("-d", "--data_source", choices=["boss", "lamost_dr7", "lamost_dr8", "apogee"], default="boss", 
                        help="source of data. Default is boss. Options: [boss, lamost_dr7, lamost_dr8, apogee]")
    
    parser.add_argument("-o", "--output_file", type=str, default=None,
                        help="Path to the file where the predictions will be saved. Default is None")

    parser.add_argument("-u", "--num_uncertainty_draws", type=int, default=0,
                        help="Number of realizations to sample from predictive distribution to calculate uncertainties. Default is 0.")
    
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Print progress messages")

    return parser.parse_args()

def main() -> None:
    """
    Main function that runs the pipeline with the specified arguments.
    """
    args = parse_args()
    pipeline = Pipeline(
        spectra_paths=args.spectra_paths,
        data_source=args.data_source,
        output_file=args.output_file,
        num_uncertainty_draws=args.num_uncertainty_draws,
        verbose=args.verbose
    ).predict()

if __name__ == "__main__":
    main()

    # /research/hutchinson/workspace/sizemol/junk/list_of_paths.txt
