import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from cudams.similarity import CudaCosineGreedy
from cudams.utils import download
from numba import cuda
from tqdm import tqdm

# Importing functions used within the script
from cudams.utils import Timer

# Constants
DEFAULT_SPECTRA_FILE = 'GNPS-LIBRARY-default-filter-nmax-1024.pickle'
DEFAULT_CHUNK_SIZES_MIN = 32
DEFAULT_CHUNK_SIZES_MAX = 10000 # Equals 
DEFAULT_NUM_EVALS = 15
DEFAULT_N_MAX_PEAKS = 1024
DEFAULT_MATCH_LIMIT = 1024
DEFAULT_TOLERANCE = 0.01
DEFAULT_BATCH_SIZE = 2048 * 2

def main(args):
    # Hardware information
    print(f'Number of CPU cores: {os.cpu_count()}')
    if cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")

    # Download or load spectra
    spectra_file = download(args.spectra_file)
    spectra = pickle.loads(Path(spectra_file).read_bytes())

    # Generate chunk sizes
    chunk_sizes_min = args.chunk_sizes_min
    chunk_sizes_max = args.chunk_sizes_max
    num_evals = args.num_evals
    chunk_sizes_cu = np.round(np.logspace(
        np.log2(chunk_sizes_min),
        np.log2(chunk_sizes_max),
        num=num_evals,
        base=2,
        endpoint=True)
    ).astype(int)

    # Benchmarking
    times = []
    pairs = []
    batch_size = args.batch_size
    match_limit = args.match_limit
    n_max_peaks = args.n_max_peaks

    kernel = CudaCosineGreedy(batch_size=batch_size,
                              n_max_peaks=n_max_peaks, 
                              match_limit=match_limit)
    kernel.matrix(spectra[:64], spectra[:64])  # Warm-up

    for chunk_size in tqdm(chunk_sizes_cu):
        chunk_size = min(len(spectra), chunk_size)  # In case we run out of spectra
        references = spectra[:chunk_size]
        queries = references  # Pairwise
        with Timer() as timer:
            kernel.matrix(references, queries)
        times.append(timer.duration)
        pairs.append(len(references) * len(queries))

    # Writing results to JSONL file
    result = {
        **(vars(args)),
        'pairs': pairs,
        'times': times,
        'device': torch.cuda.get_device_name() if cuda.is_available() else "CPU",
        'nproc': os.cpu_count(),
    }

    output_file = 'benchmark.jsonl'
    with open(output_file, 'a') as f:
        f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarking for CosineGreedy algorithm')
    parser.add_argument('--spectra-file', type=str, default=DEFAULT_SPECTRA_FILE,
                        help='Path to the spectra file (pickle format)')
    parser.add_argument('--chunk-sizes-min', type=int, default=DEFAULT_CHUNK_SIZES_MIN,
                        help='Minimum size for chunking')
    parser.add_argument('--chunk-sizes-max', type=int, default=DEFAULT_CHUNK_SIZES_MAX,
                        help='Maximum size for chunking')
    parser.add_argument('--num-evals', type=int, default=DEFAULT_NUM_EVALS,
                        help='Number of evaluations for chunk sizes')
    parser.add_argument('--n-max-peaks', type=int, default=DEFAULT_N_MAX_PEAKS,
                        help='Maximum number of peaks to retain in any spectra')
    parser.add_argument('--match-limit', type=int, default=DEFAULT_MATCH_LIMIT,
                        help='Maximum matches to accumulate')
    parser.add_argument('--tolerance', type=int, default=DEFAULT_TOLERANCE,
                        help='Maximum matches to accumulate')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Optimal batch size for hardware')
    
    args = parser.parse_args()

    main(args)
