import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from pipeline import Pipeline
from utils import filter_fits_table
import tempfile

def parse_args():
    """Parses command line arguments for running the model predictions."""
    parser = argparse.ArgumentParser(description="Run predictions on fits files and save to a file.")

    parser.add_argument("--fits_dir", type=str, required=True,
                        help="Path to the directory containing the fits files containing spectra.")
    
    parser.add_argument("--fits_table", type=str, default="./",
                        help="Path to the fits table containing paths to spectra used for predictions.")
    
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to the file where the predictions will be saved. Default is None")

    parser.add_argument("--num_uncertainty_draws", type=int, default=0,
                        help="Number of realizations to sample from predictive distribution to calculate uncertainties. Default is 0.")
    
    parser.add_argument("--filter_valid_paths", action="store_true", default=False,
                        help="Whether to filter the fits table for valid paths to the fits files. Default is False")
    
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print progress messages")
    
    return parser.parse_args()

def main() -> None:
    """
    Main function that runs the pipeline with the specified arguments.
    """
    args = parse_args()
    if args.filter_valid_paths:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_fits_table = os.path.join(tempdir, "temp_table.fits")
            filter_fits_table(args.fits_table, args.fits_dir, temp_fits_table, verbose=args.verbose)
            pipeline = Pipeline(temp_fits_table, args.fits_dir, args.output_path, num_uncertainty_draws=args.num_uncertainty_draws, verbose=args.verbose)
            pipeline.predict()
    else:
        pipeline = Pipeline(args.fits_table, args.fits_dir, args.output_path, num_uncertainty_draws=args.num_uncertainty_draws, verbose=args.verbose)
        pipeline.predict()

if __name__ == "__main__":
    main()
    

# python bossnet --filter_valid_paths --fits_table /uufs/chpc.utah.edu/common/home/u6031723/logan_files/filtered_table2.fits --fits_dir /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/healpix/ --verbose
# python bossnet --fits_table /uufs/chpc.utah.edu/common/home/u6031723/logan_files/filtered_table2.fits --fits_dir /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/spectro/healpix/ --verbose --output_path ../preds.csv