"""
Entry point to bossnet pipeline.

MIT License
Copyright (c) 2025 hutchresearch

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

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import BOSSNETPipeline, ApogeeNetPipeline, GAIARVSPipeline, GAIAXpPipeline
from utils import Stats, open_yaml, DataSource

def parse_args():
    """Parses command line arguments for running the model predictions."""
    parser = argparse.ArgumentParser(description="Run predictions on fits files and save to a file.")
    
    parser.add_argument("spectra_paths", type=str,
                        help="Path to plain text file containing paths to spectra used for predictions.")

    parser.add_argument("-d", "--data_source", choices=[e.value for e in DataSource], default="boss", 
                        help=f"source of data. Default is boss. Options: {[e.value for e in DataSource]}")
    
    parser.add_argument("-o", "--output_file", type=str, default=None,
                        help="Path to the file where the predictions will be saved. Default is None")

    parser.add_argument("-u", "--num_uncertainty_draws", type=int, default=0,
                        help="Number of realizations to sample from predictive distribution to calculate uncertainties. Default is 0.")
    
    parser.add_argument("-b", "--batch_size", type=int, default=128,
                        help="Batch size for dataloader.")
    
    parser.add_argument("-w", "--num_workers", type=int, default=6,
                        help="Number of workers for dataloader.")
    
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Print progress messages")

    return parser.parse_args()


def main() -> None:
    """
    Main function that runs the pipeline with the specified arguments.
    """
    args = parse_args()

    stats_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], "stats")
    stats_dict = open_yaml(os.path.join(stats_dir, args.data_source + ".yaml"))
    stats = Stats(**stats_dict)

    if args.data_source in [DataSource.BOSS, DataSource.LAMOSTDR7, DataSource.LAMOSTDR8]:
        pipeline = BOSSNETPipeline(
            spectra_paths=args.spectra_paths,
            data_source=args.data_source,
            stats=stats,
            output_file=args.output_file,
            num_uncertainty_draws=args.num_uncertainty_draws,
            verbose=args.verbose
        )

    elif args.data_source == DataSource.APOGEE:
        pipeline = ApogeeNetPipeline(
            spectra_paths=args.spectra_paths,
            stats=stats,
            output_file=args.output_file,
            num_uncertainty_draws=args.num_uncertainty_draws,
            verbose=args.verbose
        )

    elif args.data_source == DataSource.GAIARVS:
        pipeline = GAIARVSPipeline(
            file_path=args.spectra_paths,
            stats=stats,
            output_file=args.output_file,
            num_uncertainty_draws=args.num_uncertainty_draws,
            verbose=args.verbose,
        )
    
    elif args.data_source == DataSource.GAIAXP:
        pipeline = GAIAXpPipeline(
            file_path=args.spectra_paths,
            stats=stats,
            output_file=args.output_file,
            num_uncertainty_draws=args.num_uncertainty_draws,
            verbose=args.verbose,
        )
    
    pipeline.predict(args.batch_size, args.num_workers)
        

if __name__ == "__main__":
    main()
