import math
import warnings
from itertools import product
from pathlib import Path
from typing import List, Literal
import numpy as np
import torch
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.typing import SpectrumType
from numba import cuda
from tqdm import tqdm

from ..utils import (argbatch, spectra_peaks_to_tensor)
from .spectrum_similarity_functions import cosine_greedy_kernel


class CudaCosineGreedy(BaseSimilarity):
    score_datatype = [("score", np.float32), ("matches", np.int32), ("overflow", np.uint8)]

    def __init__(
        self,
        tolerance: float = 0.1,
        mz_power: float = 0.0,
        intensity_power: float = 1.0,
        shift: float = 0,
        batch_size: int = 1024,
        match_limit: int = 1024,
        verbose=False,
    ):
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.int_power = intensity_power
        self.shift = shift
        self.batch_size = batch_size
        self.match_limit = match_limit
        self.verbose = verbose

        self.kernel = cosine_greedy_kernel(
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
        device = torch.device('cuda')
        host = torch.device('cpu')
        batches_r = []
        for bstart, bend in argbatch(references, self.batch_size):
            rbatch = references[bstart:bend]
            rspec, rlen = spectra_peaks_to_tensor(rbatch, dtype=np.float32, ignore_null_spectra=True)
            batches_r.append([rspec, rlen, bstart, bend])

        batches_q = []
        for bstart, bend in argbatch(queries, self.batch_size):
            qbatch = queries[bstart:bend]
            qspec, qlen = spectra_peaks_to_tensor(qbatch, dtype=np.float32, ignore_null_spectra=True)
            batches_q.append([qspec, qlen, bstart, bend])

        batched_inputs = tuple(product(batches_r, batches_q))
        
        R, Q = len(references), len(queries)
        
        with torch.no_grad():
            result = torch.empty(3, R, Q, dtype=torch.float32, device=device)
            for batch_i in tqdm(range(len(batched_inputs))):
                (rspec, rlen, rstart, rend), (qspec, qlen, qstart, qend) = batched_inputs[
                    batch_i
                ]
                
                lens = torch.zeros(2, self.batch_size, dtype=torch.int32)
                lens[0, :len(rlen)] = torch.from_numpy(rlen)
                lens[1, :len(qlen)] = torch.from_numpy(qlen)
                lens = lens.to(device)
                
                rspec = torch.from_numpy(rspec).to(device)
                qspec = torch.from_numpy(qspec).to(device)
            
                rspec = cuda.as_cuda_array(rspec)
                qspec = cuda.as_cuda_array(qspec)
                lens = cuda.as_cuda_array(lens)
                
                out = torch.empty(3, self.batch_size, self.batch_size, dtype=torch.float32, device=device)
                out = cuda.as_cuda_array(out)
                
                self.kernel(
                    rspec, qspec, lens, out
                )
                
                out = torch.as_tensor(out)
                result[:, rstart:rend, qstart:qend] = out[:, :len(rlen), :len(qlen)]
        # result[:, rstart:rend, qstart:qend] = out
        # result = np.rec.fromarrays(
        #     result[:, :len(references),:len(queries)], 
        #     dtype=self.score_datatype)
        # score, matches, overflow = result.to(host)
        # return dict(
        #     score=score.float().numpy(),
        #     matches=matches.int().numpy(),
        #     overflow=overflow.bool().numpy()
        # )
        # scores, matches, overflow = result.to(host).numpy()
        # print(scores.shape, R, Q)
        return np.rec.fromarrays(
            result.to(host).numpy(),
            dtype=self.score_datatype,
        )

    def _matrix_with_sparse_output(
        self,
        references: List[Spectrum],
        queries: List[Spectrum],
        threshold: float = .75,
        is_symmetric: bool = False,
    ) -> np.ndarray:        
        if is_symmetric:
            warnings.warn("is_symmetric is ignored here, it has no effect.")
            
        if threshold <= .5 and len(references) * len(queries) > 5_000 ** 2:
            warnings.warn(f"Threshold of {threshold} when working with large spectra will likely cause an OOM error. Use with care.")
            
        batches_r = []
        for bstart, bend in argbatch(references, self.batch_size):
            rbatch = references[bstart:bend]
            rspec, rlen = spectra_peaks_to_tensor(rbatch, dtype=np.float32, ignore_null_spectra=True)
            batches_r.append([rspec, rlen, bstart, bend])

        batches_q = []
        for bstart, bend in argbatch(queries, self.batch_size):
            qbatch = queries[bstart:bend]
            qspec, qlen = spectra_peaks_to_tensor(qbatch, dtype=np.float32, ignore_null_spectra=True)
            batches_q.append([qspec, qlen, bstart, bend])

        batched_inputs = tuple(product(batches_r, batches_q))
        device = torch.device('cuda')
        host = torch.device('cpu')
        
        results = []

        with torch.no_grad():
            for batch_i in tqdm(range(len(batched_inputs)), disable=not self.verbose):
                (rspec, rlen, rstart, rend), (qspec, qlen, qstart, qend) = batched_inputs[
                    batch_i
                ]
                
                lens = torch.zeros(2, self.batch_size, dtype=torch.int32)
                lens[0, :len(rlen)] = torch.from_numpy(rlen)
                lens[1, :len(qlen)] = torch.from_numpy(qlen)
                
                lens = lens.to(device)
                
                rspec = torch.from_numpy(rspec).to(device)
                qspec = torch.from_numpy(qspec).to(device)
            
                rspec = cuda.as_cuda_array(rspec)
                qspec = cuda.as_cuda_array(qspec)
                lens = cuda.as_cuda_array(lens)
                    
                out = torch.empty(3, self.batch_size, self.batch_size, dtype=torch.float32, device=device)
                out = cuda.as_cuda_array(out)
                
                self.kernel(rspec, qspec, lens, out)
                
                out = torch.as_tensor(out, device=device)
                mask = out[0] >= threshold
                row, col = torch.nonzero(mask, as_tuple=True)
                rabs = (rstart + row).to(host)
                qabs = (qstart + col).to(host)
                score, matches, overflow = out[:, mask].to(host)
                
                results.append(
                    dict(
                        rabs=rabs.int().numpy(),
                        qabs=qabs.int().numpy(),
                        score=score.float().numpy(),
                        matches=matches.int().numpy(),
                        overflow=overflow.bool().numpy(),
                    )
                )

        rabs = []
        qabs = []
        score = []
        matches = []
        overflow = []
        for bunch in tqdm(results, disable=not self.verbose):
            qabs += [bunch['qabs']]
            rabs += [bunch['rabs']]
            score += [bunch['score']]
            matches += [bunch['matches']]
            overflow += [bunch['overflow']]
            
        qabs = np.concatenate(qabs)
        rabs = np.concatenate(rabs)
        score = np.concatenate(score)
        matches = np.concatenate(matches)
        overflow = np.concatenate(overflow)
        
        return np.rec.fromarrays(
            [
                qabs, rabs, score, matches, overflow
            ]
        )
                # np.savez_compressed(
                #     f'data/output/{rstart}-{rend}-{qstart}-{qend}.npz', 
                #     rabs=rabs.int().to(host), 
                #     qabs=qabs.int().to(host), 
                #     score=score.float(),
                #     matches=matches.int(),
                #     overflow=overflow.bool()
                # )

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