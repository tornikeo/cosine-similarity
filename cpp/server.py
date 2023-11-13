from io import BytesIO
import subprocess

from fastapi import FastAPI, File, UploadFile
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Tuple, List
from matchms import Spectrum
from matchms.typing import SpectrumType
import numpy as np
import pandas as pd
from pathlib import Path
import json

from matchms import Spectrum

from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity
from matchms.filtering import reduce_to_number_of_peaks
from matchms.filtering import add_losses

def process_spectrum(spectrum):
    spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001)
    spectrum = reduce_to_number_of_peaks(spectrum, n_max=1000)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
    return spectrum


def get_ref_spectra_from_df(spectra_df):
    # This function will take a dataframe with spectra and return a list of matchms spectra
    # Argh, This function is annoyingly slow. Added simple parallelization.
    
    # for index, row in spectra_df.iterrows():
    def fn(index, row):
        pbid = row["pbid"]
        precursor_mz = row["precursor_mz"]
        smiles = row["pb_smiles"]
        inchikey = row["pb_inchikey"]
        mz_array = np.array(json.loads(row["peaks_mz"]))
        intensity_array = np.array(json.loads(row["peaks_intensities"]))
        sp = Spectrum(mz=mz_array, intensities=intensity_array,
                        metadata={'id': pbid, 
                                'precursor_mz': precursor_mz, 
                                'smiles': smiles, 
                                'inchikey': inchikey}) 
        sp = process_spectrum(sp)
        return sp
    
    spectra = Parallel(-2)(delayed(fn)(index, row) for index, row in tqdm(spectra_df.iterrows(), total=len(spectra_df)))
    spectra = [s for s in spectra if s is not None]
    return spectra

def spectra_peaks_to_tensor(spectra: list, fill: float):
    sp_max_shape = max(len(s.peaks) for s in spectra)
    sp = np.full((len(spectra), sp_max_shape, 2), fill, 'float32')
    batch = np.zeros(len(spectra),dtype=np.uint64)
    for i, s in enumerate(spectra):
        sp[i, :len(s.peaks)] = s.peaks.to_numpy
        batch[i] = len(s.peaks)
    return sp, batch

app = FastAPI(debug=True)

@app.post('/calculate-greedy-cosine')
def create_upload_file(n_queries: int, file: UploadFile):
    contents = file.file.read()
    buffer = BytesIO(contents)
    ref_spectra_df = pd.read_csv(buffer)
    spectra = get_ref_spectra_from_df(ref_spectra_df)
    queries = spectra[:n_queries]
    references = spectra[n_queries:]
    

    # queries = large_references[:1000]
    # references = large_references[1000:]

    references_batch, _ \
        = spectra_peaks_to_tensor(references, fill=-1e6)
    queries_batch, _ \
        = spectra_peaks_to_tensor(queries, fill=-1e6)
    
    Path('data').mkdir(exist_ok=True)
    
    np.save('data/references_mz.npy', references_batch[...,0])
    np.save('data/references_int.npy', references_batch[...,1])
    np.save('data/queries_mz.npy', queries_batch[...,0])
    np.save('data/queries_int.npy', queries_batch[...,1])
    
    subprocess.call(
        "./cosine --input_dir data/ --output_dir data/".split()
    )
    
    return {"filename": file.filename}

