from pydantic import BaseModel
import warnings
# from cudams import cosine, data, utils
from cudams.utils import argbatch, mkdir
from cudams.data import get_ref_spectra_from_df
from cudams.kernel import compile
from cudams.utils import name2idx
import math
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from cudams.data import spectra_peaks_to_tensor
from numba import cuda
from itertools import product
from time import perf_counter
from multiprocessing.pool import ThreadPool
from multiprocessing import shared_memory
import numpy as np
import json

class Config(BaseModel):
    tolerance: float
    shift: float
    mz_power: int
    int_power: int
    dtype: str
    reference_csv_file: Path
    query_csv_file: Path
    BATCH_SIZE: int
    MATCH_LIMIT: int
    LIMIT: int
    
class CosineGreedyResults:
    def __init__(self, 
                 config: Config,
                 files: list[Path],
                 len_references: int, 
                 len_queries: int) -> None:
        self.files = files
        self.len_references = len_references
        self.len_queries = len_queries
        self.config = config
        assert all([f.exists() for f in self.files])
    
    def __str__(self) -> str:
        return f"CosineGreedyResults {str(self.config)}"
    
    def to_full_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        BATCH_SIZE = self.config.BATCH_SIZE
        R = math.ceil( self.len_references / BATCH_SIZE ) * BATCH_SIZE
        Q = math.ceil( self.len_queries / BATCH_SIZE ) * BATCH_SIZE

        G = np.empty((R, Q, 2), dtype='float32')
        Ov = np.empty((R, Q, 1), dtype='uint8')
        
        files = sorted([f for f in self.files])
        for file in files:
            rstart, rend, qstart, qend = name2idx(file)
            print("FILE HIT", file.stem, rstart, rend, qstart, qend)
            if 'score' in file.stem:
                # print('score', file)
                G[rstart:rend, qstart:qend] = np.load(file)
            elif 'ovfl' in file.stem:
                print('ovfl HIT', file, file.stem)
                Ov[rstart:rend, qstart:qend] = np.load(file)
        return G, Ov
    
    def to_pandas(self) -> pd.DataFrame:
        results = pd.DataFrame([], columns=['Reference','Query','Score','Num_Matches'])
        # scores = sorted(gpu_output_dir.glob('*.score.npy'))
        scores = [f for f in self.files if f.stem.endswith('score')]
        for score in scores:
            rstart, rend, qstart, qend = name2idx(score)
            score = np.load(score)
                # Condition query
            pairs_relative = np.argwhere(score[...,0] >= self.min_score)
            # We have to pad pairs with their actual locations on full grid
            pairs_absolute = pairs_relative + [rstart, qstart]
            
            # score, num_matches = get_one_specific(ref_idx, que_idx)
            r, q = pairs_relative.T
            score, num_match = score[r, q].T
            
            r, q = pairs_absolute.T
            result = pd.DataFrame(dict(
                Reference=r.astype('uint32'),
                Query=q.astype('uint32'),
                Score=score.astype('float32'),
                Num_Matches=num_match.astype('uint16')
            )).convert_dtypes()
            results = pd.concat([results, result], axis=0, copy=False)
        return results
            
    def filter(self, min_score):
        total_gbs = self.len_references * self.len_queries * 4 * 2 * 1e9
        self.min_score = min_score
        if min_score < .5 and total_gbs > 1:
            warnings.warn(f"min_score <= .5 on large results requres a LOT of RAM.")
        return self
        
class CudaCosineGreedy:
    def __init__(self, 
                 config: Config, 
                 output_dir: Path,
                 kernel: callable,
                 ) -> None:
        self.kernel = kernel
        self.config = config
        
        self.output_dir = output_dir
        self._results = None
        
    def __str__(self) -> str:
        return self.config.model_dump_json(indent=1)

    def __call__(self, references, queries) -> CosineGreedyResults:
        CONFIG = self.config
        BATCH_SIZE = CONFIG.BATCH_SIZE
        dtype = CONFIG.dtype
        
        batches_r = []
        for bstart, bend in tqdm(argbatch(references, BATCH_SIZE), desc="Batch all references"):
            rbatch = references[bstart:bend]
            rspec, rlen = spectra_peaks_to_tensor(rbatch, dtype=dtype)
            batches_r.append([rspec, rlen, bstart, bend])

        batches_q = []
        for bstart, bend in tqdm(argbatch(queries, BATCH_SIZE), desc="Batch all queries"):
            qbatch = queries[bstart:bend]
            qspec, qlen  = spectra_peaks_to_tensor(qbatch, dtype=dtype)
            batches_q.append([qspec, qlen, bstart, bend])

        batches_rq = list(product(batches_r, batches_q))
        streams = [cuda.stream() for _ in range(len(batches_rq))]

        TOTAL_BATCHES = len(batches_rq)

        gpu_output_dir = mkdir(self.output_dir, clean=True)
        
        # We initialize a pool of 3 workers that will offload results to disk
        with ThreadPool(3) as pool:
            # We loop over all batchs in sequence
            for batch_i in tqdm(range(TOTAL_BATCHES)):
                
                # Each batch has own CUDA stream so that the GPU is as busy as possible
                stream = streams[batch_i]
                
                # Shared memory allows pool workers to read array without copying it
                out_shm = shared_memory.SharedMemory(create=True, size=(BATCH_SIZE * BATCH_SIZE * 2 * 4))
                out = np.ndarray(shape=(BATCH_SIZE, BATCH_SIZE, 2), dtype='float32', buffer=out_shm.buf)
                
                overflow_shm = shared_memory.SharedMemory(create=True, size=(BATCH_SIZE * BATCH_SIZE * 1 * 1))
                overflow = np.ndarray(shape=(BATCH_SIZE, BATCH_SIZE, 1), dtype='uint8', buffer=overflow_shm.buf)

                # We get our batch and lengths (lengths are different for different spectra)
                (rspec, rlen, rstart, rend), (qspec, qlen, qstart, qend) = batches_rq[batch_i]
                lens = np.zeros((2, BATCH_SIZE), 'int32')
                lens[0,:len(rlen)] = rlen
                lens[1,:len(qlen)] = qlen
                
                # We make sure main resources remain on CPU RAM
                with cuda.pinned(rspec, qspec, lens, out, overflow,):
                    # We order empty space for results on GPU RAM
                    out_cu = cuda.device_array((BATCH_SIZE, BATCH_SIZE, 2), dtype='float32', stream=stream)
                    overflow_cu = cuda.device_array((BATCH_SIZE, BATCH_SIZE, 1), dtype='uint8', stream=stream)

                    # We order the stream to copy input data to GPU RAM
                    rspec_cu = cuda.to_device(rspec, stream=stream)
                    qspec_cu = cuda.to_device(qspec, stream=stream)
                    lens_cu = cuda.to_device(lens, stream=stream)
                    
                    # We order the stream to execute kernel (this is scheduled, it will execute, but we can't force it)
                    self.kernel(rspec_cu, qspec_cu,
                            lens_cu,
                            out_cu, overflow_cu,
                            stream=stream)
                    
                    # We order a data return
                    out_cu.copy_to_host(out, stream=stream)
                    overflow_cu.copy_to_host(overflow, stream=stream)

                    # We create a function that will execute when this stream is done working
                    # It is important to be quick here - so main work of writing to disk
                    # Is handled by pool workers, not callback stream.
                    def end_of_stream_callback(stream, status, args):
                        
                        def thread_worker(name1, name2, rstart, rend, qstart, qend):
                            ex_shm = shared_memory.SharedMemory(name=name1)
                            out = np.ndarray(shape=(BATCH_SIZE, BATCH_SIZE, 2), dtype=np.float32, buffer=ex_shm.buf)
                            np.save(gpu_output_dir / f'{rstart}-{rend}.{qstart}-{qend}.score.npy', out)
                            ex_shm.unlink()
                            
                            ex_shm = shared_memory.SharedMemory(name=name2)
                            overflow = np.ndarray(shape=(BATCH_SIZE, BATCH_SIZE, 1), dtype=np.uint8, buffer=ex_shm.buf)
                            np.save(gpu_output_dir / f'{rstart}-{rend}.{qstart}-{qend}.ovfl.npy', overflow)
                            ex_shm.unlink()
                        pool.apply_async(
                            thread_worker, 
                            args=args, 
                            error_callback=lambda e: print("Thread error", e)
                        )
                    stream.add_callback(
                        callback=end_of_stream_callback,
                        arg=[
                            out_shm.name,
                            overflow_shm.name,
                            rstart, rend, qstart, qend
                        ]
                    )
        pool.join()
        # We wait for all streams to finish their work everywhere 
        cuda.synchronize()
        
        files = sorted(list(gpu_output_dir.glob('*.npy')))
        print(files)
        print(len(files), len(batches_rq))
        assert len(batches_rq) * 2 == len(files), "Some files are missing."
        
        self._results = CosineGreedyResults(
            config=self.config,
            files=sorted(files),
            len_queries=len(queries),
            len_references=len(references),
        )
        return self._results