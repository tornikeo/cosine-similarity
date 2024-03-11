import contextlib
import io
import json
import re
import shutil
import subprocess
import sys
import warnings
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Iterable, List, Literal, Optional

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.filtering import (add_losses, normalize_intensities,
                               reduce_to_number_of_peaks,
                               require_minimum_number_of_peaks, select_by_mz,
                               select_by_relative_intensity)
from tqdm import tqdm


def argbatch(lst: list, batch_size: int) -> Iterable[tuple[int, int]]:
    """
    Given list, return `batch_size`-d chunks from it but only as indexing arguments!
    Can be used as follows:
    ```
    for bstart, bend in argbatch(references, 2048):
        rbatch = references[bstart:bend]
        # Do something with `rbatch`
    ```
    """
    for i in range(0, len(lst), batch_size):
        yield i, i + batch_size


def mkdir(p: Path, clean=False) -> Path:
    p = Path(p)
    if clean and p.is_dir() and p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(exist_ok=True, parents=True)
    return p


@contextlib.contextmanager
def mute_stdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout


def ignore_performance_warnings():
    from numba.core.errors import NumbaPerformanceWarning

    warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


def spectra_peaks_to_tensor(
    spectra: list,
    dtype: str = "float32",
    pad: int = None,
    ignore_null_spectra: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Working with GPU requires us to have a fixed shape for mz/int arrays.
    This isn't the case for real-life data, so we have to either pad or trim the spectra so that
    all of them are the same size.

    Returns:
        spectra: [2, len(spectra)] float32
        batch: [len(spectra)] int32
    """
    if pad is None:
        sp_max_shape = max(len(s.peaks) for s in spectra)
    else:
        sp_max_shape = pad

    if sp_max_shape > 2048:
        warnings.warn(
            "Some spectra are very long (>2048 peaks), consider using `matchms.filtering.reduce_to_number_of_peaks` to"
            "avoid unnecessarily long computation times."
        )

    mz = np.empty((len(spectra), sp_max_shape), dtype=dtype)
    int = np.empty((len(spectra), sp_max_shape), dtype=dtype)
    batch = np.empty(len(spectra), dtype=np.int32)
    for i, s in enumerate(spectra):
        if s is not None:
            # .to_numpy creates an unneeded copy - we don't need to do that twice
            # spec_len = min(len(s.peaks), spectra_len_cutoff)
            spec_len = min(len(s.peaks), sp_max_shape)
            mz[i, :spec_len] = s._peaks.mz[:spec_len]
            int[i, :spec_len] = s._peaks.intensities[:spec_len]
            batch[i] = spec_len
        elif s is None and ignore_null_spectra:
            batch[i] = 0
    spec = np.stack([mz, int], axis=0)
    return spec, batch


def process_spectrum(spectrum: Spectrum) -> Optional[Spectrum]:
    """
    One of the many ways to preprocess the spectrum - we use this by default.
    """
    spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001)
    spectrum = reduce_to_number_of_peaks(spectrum, n_max=1024)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
    return spectrum


def get_ref_spectra_from_df(
    spectra_df,
    limit=None,
    spectrum_processor: callable = process_spectrum,
) -> list[Spectrum]:
    """
    Convenience function that reads a pair of CSV datasets with columns like `pbid`,`precursor_mz`, etc. and
    returns parsed and preprocessed spectra as a list.
    """
    from joblib import Parallel, delayed

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
    reference_csv_file="data/input/example_dataset_tornike.csv",
    query_csv_file="data/input/example_dataset_tornike.csv",
    preprocess: Literal["minimal", "full"] = "minimal",
    max_peaks=1024,
    batch_size=512,
    max_pairs=512**2,
    padding=None,
    dtype="float32",
    verbose=False,
) -> tuple[list[Spectrum], list[Spectrum], list[np.ndarray]]:
    """
    Convenience function that does everything the `get_ref_spectra_from_df` does, but also
    transforms all the spectra into neat batches with same shape, so that they are ready for
    GPU processing.

    :param str reference_csv_file: A suitable csv file path (str)
    :param str query_csv_file: A suitable csv file path (str)
    :param str preprocess: Can be 'minimal' or 'full'. Determines which `matchms.filtering` functions we will use for spectra.
    :param int max_peaks: determines the maximum length of spectra (after this number, spectra are truncated, if spectra are smaller, they are instead padded with zeros).
    :param int padding: unused - this would make every *batch* the same shape. For current kernels we don't need this. If we were to port this to support Google's TPUs, we will likely need this then.
    :param str dtype: numpy dtype of returned batches.
    :param bool verbose: Allows reporting progress using `tqdm`.
    :return: three lists - references, queries and list of numpy matrices. The latter can be
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
        spectrum = reduce_to_number_of_peaks(spectrum, n_max=max_peaks)
        return spectrum

    process_spectrum = (
        process_spectrum_full if preprocess == "full" else process_spectrum_minimal
    )

    limit = None
    if max_pairs is not None:
        limit = int(max_pairs**0.5) * 2

    ref_spectra_df_path = Path(reference_csv_file)
    ref_spectra_df = pd.read_csv(ref_spectra_df_path)
    references = get_ref_spectra_from_df(
        ref_spectra_df, spectrum_processor=process_spectrum, limit=limit
    )

    if reference_csv_file == query_csv_file:
        queries = references[:]
    else:
        query_spectra_df_path = Path(query_csv_file)
        query_spectra_df = pd.read_csv(query_spectra_df_path)
        queries = get_ref_spectra_from_df(
            query_spectra_df,
            spectrum_processor=process_spectrum,
            limit=limit,
        )

    if max_pairs is not None:
        references = references[: int(max_pairs**0.5)]
        queries = queries[: int(max_pairs**0.5)]

    batches_r = []
    for bstart, bend in tqdm(
        argbatch(references, batch_size),
        desc="Batch all references",
        disable=not verbose,
    ):
        rbatch = references[bstart:bend]
        rspec, rlen = spectra_peaks_to_tensor(rbatch, dtype=dtype, pad=padding)
        batches_r.append([rspec, rlen, bstart, bend])

    batches_q = []
    for bstart, bend in tqdm(
        argbatch(queries, batch_size),
        desc="Batch all queries",
        disable=not verbose,
    ):
        qbatch = queries[bstart:bend]
        qspec, qlen = spectra_peaks_to_tensor(qbatch, dtype=dtype, pad=padding)
        batches_q.append([qspec, qlen, bstart, bend])

    batches_inputs = list(product(batches_r, batches_q))

    return references, queries, batches_inputs


def download(
    name: Literal[
        "ALL_GNPS.mgf",
        "ALL_GNPS.pickle",
        "GNPS-LIBRARY.mgf",
        "GNPS-LIBRARY.pickle",
        "pesticides.mgf",
        "GNPS-MSMLS.mgf",
        "MASSBANK.mgf",
    ]
) -> str:
    import pooch

    if "ALL_GNPS" in name:
        warnings.warn(
            f"As of 2024, {name} is a large file (1.76GB) make sure the machine can handle this"
        )

    known_hash = {
        "GNPS-LIBRARY.mgf": "235f7518536e4b04a4fd11def3d60cffada2760ceee8e96ca356d873dbb2b440",
        "ALL_GNPS.mgf": "22dcd7a254393688936d6c530abc3927be3bb0997125bd54dc94eff8b65947f8",
    }

    return pooch.retrieve(
        url=f"https://github.com/tornikeo/cosine-similarity/releases/download/samples-0.1/{name}",
        known_hash=known_hash.get(name),
        progressbar=True,
    )


import time


class Timer:
    def __enter__(self):
        self.duration = -time.perf_counter()
        return self

    def __exit__(self, *args):
        self.duration += time.perf_counter()
