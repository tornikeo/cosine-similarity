import json
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from matchms import Spectrum
from tqdm import tqdm
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity
from matchms.filtering import reduce_to_number_of_peaks
from matchms.filtering import add_losses
import warnings


def spectra_peaks_to_tensor(
    spectra: list, dtype: str = "float32", 
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
    # spectra_len_cutoff: int = 1024,
    sp_max_shape = max(len(s.peaks) for s in spectra)
    # if sp_max_shape > spectra_len_cutoff:
    #     print(
    #         f"""
    #         When batching peaks into a tensor, encountered an extremely long spectrum (length {sp_max_shape}).
    #         It is not yet possible to fit spectrum this long inside a batch due to memory contstraints. We will
    #         only use first {spectra_len_cutoff} elements from spectra like these.
    #         """
    #     )
    # sp_max_shape = min(sp_max_shape, spectra_len_cutoff)
    # sp_max_shape = min(sp_max_shape, spectra_len_cutoff)
    mz = np.empty((len(spectra), sp_max_shape), dtype=dtype)
    int = np.empty((len(spectra), sp_max_shape), dtype=dtype)
    batch = np.empty(len(spectra), dtype=np.int32)
    for i, s in enumerate(spectra):
        # .to_numpy creates an unneeded copy - we don't need to do that twice
        # spec_len = min(len(s.peaks), spectra_len_cutoff)
        spec_len = len(s.peaks)
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

