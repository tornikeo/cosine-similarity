import math
import warnings
from itertools import product
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from numba import cuda
from ..utils import CudaTimer
from tqdm import tqdm

from ..utils import argbatch
from .spectrum_similarity_functions import cosine_greedy_kernel


class CudaCosineGreedy(BaseSimilarity):
    score_datatype = [
        ("score", np.float32),
        ("matches", np.int32),
        ("overflow", np.uint8),
    ]

    def __init__(
        self,
        tolerance: float = 0.1,
        mz_power: float = 0.0,
        intensity_power: float = 1.0,
        shift: float = 0,
        batch_size: int = 2048,
        n_max_peaks: int = 1024,
        match_limit: int = 2048,
        verbose=False,
    ):
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.int_power = intensity_power
        self.shift = shift
        self.batch_size = batch_size
        self.match_limit = match_limit
        self.verbose = verbose
        self.n_max_peaks = n_max_peaks
        self.kernel_time = 0 # Used for debugging and timing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.kernel = cosine_greedy_kernel(
            tolerance=self.tolerance,
            shift=self.shift,
            mz_power=self.mz_power,
            int_power=self.int_power,
            match_limit=self.match_limit,
            batch_size=self.batch_size,
            n_max_peaks=self.n_max_peaks
        )
        if not cuda.is_available():
            warnings.warn(f"{self.__class__}: CUDA device seems unavailable.")

    def __str__(self) -> str:
        return self.config.model_dump_json(indent=1)

    def _spectra_peaks_to_tensor(
        self, 
        spectra: list,
        n_max_peaks: int = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        dynamic_shape = max(len(s.peaks) for s in spectra)
        n_max_peaks = dynamic_shape if n_max_peaks is None else n_max_peaks 
        
        dtype = self.score_datatype[0][1]

        mz = np.empty((len(spectra), n_max_peaks), dtype=dtype)
        int = np.empty((len(spectra), n_max_peaks), dtype=dtype)
        spectra_lens = np.empty(len(spectra), dtype=np.int32)
        for i, s in enumerate(spectra):
            if s is not None:
                # .to_numpy creates an unneeded copy - we don't need to do that twice
                spec_len = min(len(s.peaks), n_max_peaks)
                mz[i, :spec_len] = s._peaks.mz[:spec_len]
                int[i, :spec_len] = s._peaks.intensities[:spec_len]
                spectra_lens[i] = spec_len
        stacked_spectra = np.stack([mz, int], axis=0)
        return stacked_spectra, spectra_lens

    def _get_batches(self, references, queries):
        batches_r = []
        for bstart, bend in argbatch(references, self.batch_size):
            rspec, rlen = self._spectra_peaks_to_tensor(references[bstart:bend])
            batches_r.append([rspec, rlen, bstart, bend])

        batches_q = []
        for bstart, bend in argbatch(queries, self.batch_size):
            qspec, qlen = self._spectra_peaks_to_tensor(queries[bstart:bend])
            batches_q.append([qspec, qlen, bstart, bend])

        batched_inputs = tuple(product(batches_r, batches_q))
        return batched_inputs

    def pair(self, reference: Spectrum, query: Spectrum) -> float:
        result_mat = self.matrix([reference], [query])
        return np.asarray(result_mat.squeeze(), dtype=self.score_datatype)

    def matrix(
        self,
        references: List[Spectrum],
        queries: List[Spectrum],
        array_type: Literal["numpy", "sparse", "memmap"] = "numpy",
        is_symmetric: bool = False,
        score_threshold: float = None,
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
        if is_symmetric:
            warnings.warn("is_symmetric is ignored here, it has no effect.")

        if array_type == 'sparse':
            assert score_threshold is not None, "When using array_type `sparse` you have to set `score_threshold`. Scores below this will get discarded."
            assert 0 <= score_threshold <= 1, "Score threshold must be in [0, 1] range."
        

        batched_inputs = self._get_batches(references=references, queries=queries)
        R, Q = len(references), len(queries)
        timer = CudaTimer()

        if array_type == "numpy":
            result = torch.empty(3, R, Q, dtype=torch.float32, device=self.device)
        elif array_type == "sparse":
            results = []

        self.kernel_time = 0
        with torch.no_grad():
            for batch_i in tqdm(range(len(batched_inputs)), disable=not self.verbose):
                (rspec, rlen, rstart, rend), (
                    qspec,
                    qlen,
                    qstart,
                    qend,
                ) = batched_inputs[batch_i]

                lens = torch.zeros(2, self.batch_size, dtype=torch.int32)
                lens[0, :len(rlen)] = torch.from_numpy(rlen)
                lens[1, :len(qlen)] = torch.from_numpy(qlen)
                lens = lens.to(self.device)
 
                rspec = torch.from_numpy(rspec).to(self.device)
                qspec = torch.from_numpy(qspec).to(self.device)

                rspec = cuda.as_cuda_array(rspec)
                qspec = cuda.as_cuda_array(qspec)
                lens = cuda.as_cuda_array(lens)

                out = torch.empty(
                    3,
                    self.batch_size,
                    self.batch_size,
                    dtype=torch.float32,
                    device=self.device,
                )
                out = cuda.as_cuda_array(out)

                timer.start()
                self.kernel(rspec, qspec, lens, out)
                timer.stop()
                self.kernel_time += timer.elapsed_seconds() / len(batched_inputs)
                out = torch.as_tensor(out)

                if array_type == 'numpy':
                    result[:, rstart:rend, qstart:qend] = out[:, :len(rlen), :len(qlen)]
                elif array_type == 'sparse':
                    mask = out[0] >= score_threshold
                    row, col = torch.nonzero(mask, as_tuple=True)
                    rabs = (rstart + row).cpu()
                    qabs = (qstart + col).cpu()
                    score, matches, overflow = out[:, mask].to(self.device)
                    results.append(
                        dict(
                            rabs=rabs.int().cpu().numpy(),
                            qabs=qabs.int().cpu().numpy(),
                            score=score.float().cpu().numpy(),
                            matches=matches.int().cpu().numpy(),
                            overflow=overflow.bool().cpu().numpy(),
                        )
                    )

            if array_type == 'numpy':
                return np.rec.fromarrays(
                    result.cpu().numpy(),
                    dtype=self.score_datatype,
                )
            elif array_type == 'sparse':
                result = []
                for bunch in tqdm(results, disable=not self.verbose):
                    result.append((
                        bunch["qabs"],
                        bunch["rabs"],
                        bunch["score"],
                        bunch["matches"],
                        bunch["overflow"]
                    ))

                result = np.rec.fromarrays(np.concatenate(result, axis=1))

                # rabs = []
                # qabs = []
                # score = []
                # matches = []
                # overflow = []
                # for bunch in tqdm(results, disable=not self.verbose):
                #     qabs += [bunch["qabs"]]
                #     rabs += [bunch["rabs"]]
                #     score += [bunch["score"]]
                #     matches += [bunch["matches"]]
                #     overflow += [bunch["overflow"]]

                # qabs = np.concatenate(qabs)
                # rabs = np.concatenate(rabs)
                # score = np.concatenate(score)
                # matches = np.concatenate(matches)
                # overflow = np.concatenate(overflow)
                # return np.rec.fromarrays([qabs, rabs, score, matches, overflow])
