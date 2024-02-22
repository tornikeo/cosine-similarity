import math
import warnings
from itertools import product
from multiprocessing import shared_memory
from multiprocessing.pool import ThreadPool
from pathlib import Path
from time import perf_counter
from typing import List, Literal, Tuple
from matchms.typing import SpectrumType
import numpy as np
from matchms.similarity import CosineGreedy
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms import Spectrum
from numba import cuda
from pydantic import BaseModel, Field
from scipy import sparse
from tqdm import tqdm

from ..utils import get_ref_spectra_from_df, spectra_peaks_to_tensor, \
    argbatch, mkdir, name2idx
from .kernels import compile_cuda_cosine_greedy_kernel

class CudaCosineGreedy(BaseSimilarity):
    score_datatype = [("score", np.float32), ("matches", "int"), ("overflow", "uint8")]

    def __init__(
        self,
        tolerance: float = 0.1,
        mz_power: float = 0.0,
        intensity_power: float = 1.0,
        shift: float = 0,
        batch_size: int = 1024,
        match_limit: int = 512,
        verbose=False,
    ):
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.int_power = intensity_power
        self.shift = shift
        self.batch_size = batch_size
        self.match_limit = match_limit
        self.verbose = verbose

        self.kernel = compile_cuda_cosine_greedy_kernel(
            tolerance=self.tolerance,
            shift=self.shift,
            mz_power=self.mz_power,
            int_power=self.int_power,
            match_limit=self.match_limit,
            batch_size=self.batch_size,
        )
        if not cuda.is_available():
            warnings.warn(f"{self.__class__}: CUDA device seems unavailable.")

    def __str__(self) -> str:
        return self.config.model_dump_json(indent=1)

    def _matrix_with_numpy_output(
        self,
        references: List[Spectrum],
        queries: List[Spectrum],
        is_symmetric: bool = False,
    ) -> np.ndarray:
        if is_symmetric:
            warnings.warn("is_symmetric is ignored here, it has no effect.")

        batches_r = []
        for bstart, bend in argbatch(references, self.batch_size):
            rbatch = references[bstart:bend]
            rspec, rlen = spectra_peaks_to_tensor(rbatch, dtype=np.float32)
            batches_r.append([rspec, rlen, bstart, bend])

        batches_q = []
        for bstart, bend in argbatch(queries, self.batch_size):
            qbatch = queries[bstart:bend]
            qspec, qlen = spectra_peaks_to_tensor(qbatch, dtype=np.float32)
            batches_q.append([qspec, qlen, bstart, bend])

        batches_rq = list(product(batches_r, batches_q))

        TOTAL_BATCHES = len(batches_rq)

        R = math.ceil(len(references) / self.batch_size) * self.batch_size
        Q = math.ceil(len(queries) / self.batch_size) * self.batch_size
        
        # result_output = np.empty((R, Q, 2), dtype="float32")
        # result_overflow = np.empty((R, Q, 1), dtype="uint8")
        result = np.empty([3, R, Q],'float32')
        # We loop over all batchs in sequence
        for batch_i in tqdm(range(TOTAL_BATCHES), disable=not self.verbose):
            # Each batch has own CUDA stream so that the GPU is as busy as possible

            out = np.empty(
                shape=(3, self.batch_size, self.batch_size),
                dtype=np.float32,
            )

            # We get our batch and lengths (lengths are different for different spectra)
            (rspec, rlen, rstart, rend), (qspec, qlen, qstart, qend) = batches_rq[
                batch_i
            ]
            lens = np.zeros((2, self.batch_size), np.int32)
            lens[0, :len(rlen)] = rlen
            lens[1, :len(qlen)] = qlen

            # We make sure main resources remain on CPU RAM
            with cuda.pinned(
                rspec,
                qspec,
                lens,
                out,
            ):

                # We order the stream to copy input data to GPU RAM
                rspec_cu = cuda.to_device(rspec)
                qspec_cu = cuda.to_device(qspec)
                lens_cu = cuda.to_device(lens)

                # We order empty space for results on GPU RAM
                out_cu = cuda.device_array_like(out)
                
                # We order the stream to execute kernel (this is scheduled, it will execute, but we can't force it)
                self.kernel(
                    rspec_cu, qspec_cu, lens_cu, out_cu,
                )

                # We order a data return
                out_cu.copy_to_host(out)
                
                # We wait for all streams to finish their work everywhere
                cuda.synchronize()

                result[:, rstart:rend, qstart:qend] = out
        result = np.rec.fromarrays(
            result[:, :len(references),:len(queries)], 
            dtype=self.score_datatype)
        return result

    def _matrix_with_sparse_output(
        self,
        references: List[Spectrum],
        queries: List[Spectrum],
        threshold: float = .75,
        is_symmetric: bool = False,
    ) -> np.ndarray:
        batches_r = []
        for bstart, bend in tqdm(
            argbatch(references, self.batch_size), desc="Batch all references"
        ):
            rbatch = references[bstart:bend]
            rspec, rlen = spectra_peaks_to_tensor(rbatch, dtype=np.float32)
            batches_r.append([rspec, rlen, bstart, bend])

        batches_q = []
        for bstart, bend in tqdm(
            argbatch(queries, self.batch_size), desc="Batch all queries"
        ):
            qbatch = queries[bstart:bend]
            qspec, qlen = spectra_peaks_to_tensor(qbatch, dtype=np.float32)
            batches_q.append([qspec, qlen, bstart, bend])

        batches_rq = list(product(batches_r, batches_q))
        
        TOTAL_BATCHES = len(batches_rq)

        R = math.ceil(len(references) / self.batch_size) * self.batch_size
        Q = math.ceil(len(queries) / self.batch_size) * self.batch_size
        
        batch_outputs = np.empty(shape=(TOTAL_BATCHES,4),dtype=object)
        streams = [cuda.stream() for _ in range(TOTAL_BATCHES)]
        # result_output = np.empty((R, Q, 2), dtype="float32")
        # result_overflow = np.empty((R, Q, 1), dtype="uint8")

        # We loop over all batchs in sequence
        for batch_i in tqdm(range(TOTAL_BATCHES)):
            stream = streams[batch_i]
            # Each batch has own CUDA stream so that the GPU is as busy as possible
            out = np.empty(
                shape=(self.batch_size, self.batch_size, 2),
                dtype="float32",
            )

            overflow = np.empty(
                shape=(self.batch_size, self.batch_size, 1),
                dtype="uint8",
            )

            # We get our batch and lengths (lengths are different for different spectra)
            (rspec, rlen, rstart, rend), (qspec, qlen, qstart, qend) = batches_rq[
                batch_i
            ]
            lens = np.zeros((2, self.batch_size), "int32")
            lens[0, : len(rlen)] = rlen
            lens[1, : len(qlen)] = qlen

            # We make sure main resources remain on CPU RAM
            with cuda.pinned(
                rspec,
                qspec,
                lens,
                out,
                overflow,
            ):
                # We order empty space for results on GPU RAM
                out_cu = cuda.device_array(
                    (self.batch_size, self.batch_size, 2), dtype="float32",
                    stream=stream
                )
                overflow_cu = cuda.device_array(
                    (self.batch_size, self.batch_size, 1), dtype="uint8",
                    stream=stream
                )

                # We order the stream to copy input data to GPU RAM
                rspec_cu = cuda.to_device(rspec, stream=stream)
                qspec_cu = cuda.to_device(qspec, stream=stream)
                lens_cu = cuda.to_device(lens, stream=stream)

                # We order the stream to execute kernel (this is scheduled, it will execute, but we can't force it)
                self.kernel(
                    rspec_cu, qspec_cu, lens_cu, out_cu, overflow_cu,
                    stream=stream
                )

                # result_output[rstart:rend, qstart:qend] = out
                # result_overflow[rstart:rend, qstart:qend] = overflow
                def end_of_stream_callback(
                        stream, status, 
                        rstart,
                        rend,
                        qstart,
                        qend):
                    
                    # We order a data return
                    
                    out = out_cu.copy_to_host(stream=stream)
                    overflow = overflow_cu.copy_to_host(stream=stream)
                    lens = lens_cu.copy_to_host(stream=stream)
                    
                    mask = out[:len(rlen),:len(qlen),0] >= threshold
                    # r, c = np.nonzero(mask)
                    # out = out[r,c]
                    # overflow = overflow[r,c]
                    # r += rstart
                    # c += qstart
                    # batch_outputs[batch_i] = r, c, out, overflow

                stream.add_callback(
                    callback=end_of_stream_callback,
                    arg=[
                        rstart,
                        rend,
                        qstart,
                        qend,
                    ],
                )
                # We wait for all streams to finish their work everywhere
            cuda.synchronize()
        
        # rows = np.concatenate(batch_outputs[:,0],axis=0)
        # cols = np.concatenate(batch_outputs[:,1],axis=0)
        # out = np.concatenate(batch_outputs[:,2],axis=0)
        # overflows = np.concatenate(batch_outputs[:,3],axis=0)
        
        # result_num_matches = sparse.coo_matrix((result_score, (result_i, result_j)), shape=(R,Q))
        # result_overflow = sparse.coo_matrix((result_overflow, (result_i, result_j)), shape=(R,Q))
        return None
        # return rows, cols, out, overflows

    def pair(self, reference: Spectrum, query: Spectrum) -> float:
        result_mat = self.matrix([reference], [query])
        return np.asarray(result_mat.squeeze(), dtype=self.score_datatype)
    
    def matrix(
        self,
        references: List[Spectrum],
        queries: List[Spectrum],
        array_type: Literal["numpy", "sparse", "memmap"] = "numpy",
        is_symmetric: bool = False,
        sparse_threshold: float = .75
    ) -> np.ndarray:
        """Provide optimized method to calculate an np.array of similarity scores
        for given reference and query spectrums.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        array_type
            Specify the output array type. Can be "numpy", or "memmap"
        is_symmetric
            Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
            comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
            2x faster.
        threshold:
            Useful when using `array_type=sparse` and very large number of spectra. All scores < threshold are set to 0
            and the resulting large sparse score matrix is returned as a scipy.sparse matrix (both scores and overflows)
        """
        assert self.kernel is not None, "Kernel isn't compiled - use .compile() first"
        
        if array_type == "numpy":
            return self._matrix_with_numpy_output(references, queries, is_symmetric=is_symmetric)
        elif array_type == "sparse":
            return self._matrix_with_sparse_output(references, queries, is_symmetric=is_symmetric, threshold=sparse_threshold)