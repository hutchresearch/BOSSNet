import os
import io
from dataclasses import dataclass
from typing import Optional
from astropy.io import fits
from tqdm import tqdm
import yaml

# Dataclass to hold metadata stats.
@dataclass(frozen=True)
class DataStats:
    """
    A dataclass that holds metadata statistics related to a single value.

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


def open_yaml(path: str) -> dict:
    with open(os.path.expanduser(path), 'r') as handle:
        return yaml.safe_load(handle)
    
def stats_from_dict(dictionary: dict, data_class: dataclass) -> dataclass:
    data_stats_dict = {}
    for k, v in dictionary.items():
        data_stats_dict[k] = DataStats(**v)
    return data_class(**data_stats_dict)


# Utility functions not used in the pipeline, but included for completeness.
def save_chopped_file(load_path: str, save_path: str, chunks: int):
    """Splits a file into multiple chunks and saves them to disk.

    Args:
        load_path: str, The path to the input file to be split.
        save_path: str, The directory where the split files will be saved.
        chunks: int, The number of chunks to split the input file into.

    Returns:
        None: This function does not return anything, but saves the split files to disk.

    Raises:
        ValueError: If chunks is less than 1 or if either load_path or save_path are invalid.

    """
    # Check for valid inputs
    if chunks < 1:
        raise ValueError("Number of chunks must be at least 1.")
    if not os.path.isfile(load_path):
        raise ValueError("Invalid input file path.")
    if not os.path.isdir(save_path):
        raise ValueError("Invalid save directory path.")

    def save_chunk_single(save_name: str, file_in: io.BufferedReader, chunk_size: int):
        """Saves a single chunk of the input file to disk.

        Args:
            save_name: str, The filename to save the chunk as.
            file_in: io.BufferedReader, An open file object pointing to the input file.
            chunk_size: int, The size of the chunk to read from the input file.

        Returns:
            None: This function does not return anything, but saves the chunk to disk.

        """
        with open(save_name, "wb") as file_out:
            file_out.write(file_in.read(chunk_size))
    
    with open(load_path, "rb") as file_in:
        size = file_in.seek(0, 2)
        file_in.seek(0, 0)

        chunk_size = size // chunks
        # The last chunk has to be larger in cases where the filesize is not evenly divisible by the number of  chunks
        last_chunk_size = chunk_size + (size - chunk_size*chunks)

        for i in range(chunks-1):
            save_name = os.path.join(save_path, f"model_chunk_{i}")
            save_chunk_single(save_name, file_in, chunk_size)

        save_name = os.path.join(save_path, f"model_chunk_{chunks-1}")
        save_chunk_single(save_name, file_in, last_chunk_size)


def filter_fits_table(table_path: str, fits_files_path: str, save_path: str, verbose: Optional[bool] = False) -> None:
    """
    Filter a FITS table by checking the validity of the FITS files in the table.

    Args:
        table_path: str, The path to the FITS table to filter.
        fits_files_path: str The path to the directory containing the FITS files referenced in the table.
        save_path: str, The path to save the filtered FITS table.
        verbose: bool, If True, print progress information to the console. Defaults to False.

    Raises:
        OSError: If the file specified by save_path already exists.

    Returns:
        None
    """
    if os.path.exists(save_path):
        raise OSError(f"File {save_path!r} already exists.")
    
    with fits.open(table_path) as hdul:
        table = hdul[1]
        paths = table.data["HEALPIX_PATH"]

        valid_ids = []

        paths_generator = tqdm(enumerate(paths), total=len(paths), desc="Generating Filtered Table: ", ncols=80) if verbose else enumerate(paths)
        for i, path in paths_generator:
            path = os.path.join(fits_files_path, path[len("$MWM_HEALPIX/"):])
            try:
                with fits.open(path) as fits_file:
                    fits_file[0].header  # Access the header to check if the file is a valid FITS file
                valid_ids.append(i)
            except (OSError, TypeError):
                if verbose:
                    print(path) # Ignore invalid or corrupt files

        valid_data = table.data[valid_ids]
        table.data = valid_data
        table.writeto(save_path)