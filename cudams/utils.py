import contextlib
import io
import os
import re
import shutil
import sys
import warnings
from pathlib import Path
import os    
import json
import warnings
from pathlib import Path
from typing import List, Literal, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from itertools import product
from matchms import Spectrum
from matchms.filtering import (add_losses, normalize_intensities,
                               reduce_to_number_of_peaks,
                               require_minimum_number_of_peaks, select_by_mz,
                               select_by_relative_intensity)
from tqdm import tqdm
from contextlib import contextmanager
from pathlib import Path
import requests
import shutil

def batches(lst, batch_size) -> list:
    """
    Batch data from the iterable into tuples of length n. The last batch may be shorter than n.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def argbatch(lst, batch_size) -> tuple[int, int]:
    """
    Batch data from the iterable into tuples of start-end indices
    """
    for i in range(0, len(lst), batch_size):
        yield i, i + batch_size


def mkdir(p: Path, clean=False) -> Path:
    p = Path(p)
    if clean and p.is_dir() and p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(exist_ok=True, parents=True)
    return p


def name2idx(p: Path) -> tuple[int, int, int, int]:
    match = re.match(r"(\d+)-(\d+)\.(\d+)-(\d+)", p.stem)
    rstart, rend, qstart, qend = map(int, match.groups())
    return rstart, rend, qstart, qend

@contextlib.contextmanager
def mute_stdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout
    
def ignore_performance_warnings():
    from numba.core.errors import NumbaPerformanceWarning
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


def spectra_peaks_to_tensor(
    spectra: list, dtype: str = "float32", 
    pad: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Working with GPU requires us to have a fixed shape for mz/int arrays.
    This isn't the case for real-life data, so we have to "pad" the mz/int arrays.
    We keep the real size of the mz/int in separate array, "batch". The regions out
    of what "batch" specifies is undefined.

    Returns:
        spectra: [2, len(spectra)] float32
        batch: [len(spectra)] int32
    """
    if pad is None:
        sp_max_shape = max(len(s.peaks) for s in spectra)
    else:
        sp_max_shape = pad
        
    mz = np.empty((len(spectra), sp_max_shape), dtype=dtype)
    int = np.empty((len(spectra), sp_max_shape), dtype=dtype)
    batch = np.empty(len(spectra), dtype=np.int32)
    for i, s in enumerate(spectra):
        # .to_numpy creates an unneeded copy - we don't need to do that twice
        # spec_len = min(len(s.peaks), spectra_len_cutoff)
        spec_len = min(len(s.peaks), sp_max_shape)
        mz[i, :spec_len] = s._peaks.mz[:spec_len]
        int[i, :spec_len] = s._peaks.intensities[:spec_len]
        batch[i] = spec_len
    spec = np.stack([mz, int], axis=0)
    return spec, batch


def process_spectrum(spectrum: np.ndarray) -> np.ndarray:
    spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001)
    spectrum = reduce_to_number_of_peaks(spectrum, n_max=1000)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
    return spectrum

def get_ref_spectra_from_df(spectra_df, 
                            limit=None,
                            spectrum_processor: callable = process_spectrum,
                            ) -> pd.DataFrame:
    """
    This function will take a dataframe with spectra and return a list of matchms spectra.
    Since all rows are independent, this function does this preprocessing in parallel (CPU).
    """
    # for index, row in spectra_df.iterrows():
    def fn(index, row):
        pbid = row["pbid"]
        precursor_mz = row["precursor_mz"]
        smiles = row["pb_smiles"]
        inchikey = row["pb_inchikey"]
        mz_array = np.array(json.loads(row["peaks_mz"]))
        intensity_array = np.array(json.loads(row["peaks_intensities"]))
        sp = Spectrum(
            mz=mz_array,
            intensities=intensity_array,
            metadata={
                "id": pbid,
                "precursor_mz": precursor_mz,
                "smiles": smiles,
                "inchikey": inchikey,
            },
        )
        if spectrum_processor is not None:
            sp = spectrum_processor(sp)
        return sp

    if limit is not None:
        spectra_df = spectra_df.head(limit)
    spectra = Parallel(-2)(
        delayed(fn)(index, row)
        for index, row in tqdm(spectra_df.iterrows(), total=len(spectra_df))
    )
    spectra = [s for s in spectra if s is not None]
    return spectra


def get_spectra_batches(
    reference_csv_file = 'data/input/example_dataset_tornike.csv',
    query_csv_file = 'data/input/example_dataset_tornike.csv',
    preprocess: Literal['minimal','full']='minimal',
    max_peaks=1024,
    batch_size = 512,
    max_pairs = 512 ** 2,
    padding=None,
    dtype='float32',
    verbose=False,
) -> [list, list, list]:
    """
    Returns references, queries and batched inputs, ready to be used in a kernel.
    
    """
    reference_csv_file = Path(reference_csv_file)
    query_csv_file = Path(query_csv_file)
    
    def process_spectrum_full(spectrum: np.ndarray) -> np.ndarray:
        spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
        spectrum = normalize_intensities(spectrum)
        spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001)
        spectrum = reduce_to_number_of_peaks(spectrum, n_max=max_peaks)
        spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
        return spectrum
    
    def process_spectrum_minimal(spectrum: np.ndarray) -> np.ndarray:
        spectrum = reduce_to_number_of_peaks(spectrum, n_max=max_pairs)
        return spectrum

    process_spectrum = \
        process_spectrum_full if preprocess == 'full' else process_spectrum_minimal
    
    limit = None
    if max_pairs is not None:
        limit = int(max_pairs ** .5) * 2
    
    ref_spectra_df_path = Path(reference_csv_file)
    ref_spectra_df = pd.read_csv(ref_spectra_df_path)
    references = get_ref_spectra_from_df(ref_spectra_df, 
                                         spectrum_processor=process_spectrum,
                                         limit=limit)

    if reference_csv_file == query_csv_file:
        queries = references[:]
    else:
        query_spectra_df_path = Path(query_csv_file)
        query_spectra_df = pd.read_csv(query_spectra_df_path)
        queries = get_ref_spectra_from_df(query_spectra_df, 
                                          spectrum_processor=process_spectrum,
                                          limit=limit,
                                          )
    
    if max_pairs is not None:
        references = references[:int(max_pairs**.5)]
        queries = queries[:int(max_pairs**.5)]
            
    batches_r = []
    for bstart, bend in tqdm(
        argbatch(references, batch_size), desc="Batch all references", 
        disable=not verbose,
    ):
        rbatch = references[bstart:bend]
        rspec, rlen = spectra_peaks_to_tensor(rbatch, dtype=dtype, pad=padding)
        batches_r.append([rspec, rlen, bstart, bend])

    batches_q = []
    for bstart, bend in tqdm(
        argbatch(queries, batch_size), desc="Batch all queries",
        disable=not verbose,
    ):
        qbatch = queries[bstart:bend]
        qspec, qlen = spectra_peaks_to_tensor(qbatch, dtype=dtype, pad=padding)
        batches_q.append([qspec, qlen, bstart, bend])
    
    batches_inputs = list(product(batches_r, batches_q))
    
    return references, queries, batches_inputs

def download_cosine_10k_sample(path: Path) -> Path:
    path = Path(path)
    url = 'https://github.com/tornikeo/cosine-similarity/releases/download/samples-0.1/test_set_cosine.csv'
    
    # Ensure path ends with .csv
    if not path.suffix == '.csv':
        raise ValueError("Path should end with .csv")
    
    # Download the file to a temporary location
    tmp_path = path.with_suffix('.tmp')
    with requests.get(url, stream=True) as response:
        with open(tmp_path, 'wb') as tmp_file:
            shutil.copyfileobj(response.raw, tmp_file)
    
    # Move the temporary file to the desired location
    shutil.move(tmp_path, path)
    
    return path


def download_cosine_10k_sample(path: Path) -> Path:
    path = Path(path)
    url = 'https://github.com/tornikeo/cosine-similarity/releases/download/samples-0.1/spectra_10k.csv'
    
    # Ensure path ends with .csv
    if not path.suffix == '.csv':
        raise ValueError("Path should end with .csv")
    
    # Download the file to a temporary location
    tmp_path = path.with_suffix('.tmp')
    with requests.get(url, stream=True) as response:
        with open(tmp_path, 'wb') as tmp_file:
            shutil.copyfileobj(response.raw, tmp_file)
    
    # Move the temporary file to the desired location
    shutil.move(tmp_path, path)
    
    return path

def download_cosine_100k_sample(path: Path) -> Path:
    path = Path(path)
    url = 'https://github.com/tornikeo/cosine-similarity/releases/download/samples-0.1/spectra_100k.csv'
    
    # Ensure path ends with .csv
    if not path.suffix == '.csv':
        raise ValueError("Path should end with .csv")
    
    # Download the file to a temporary location
    tmp_path = path.with_suffix('.tmp')
    with requests.get(url, stream=True) as response:
        with open(tmp_path, 'wb') as tmp_file:
            shutil.copyfileobj(response.raw, tmp_file)
    
    # Move the temporary file to the desired location
    shutil.move(tmp_path, path)
    
    return path
