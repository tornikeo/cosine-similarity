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
from tqdm import tqdm

from ..utils import (argbatch, spectra_peaks_to_tensor)
from .spectrum_similarity_functions import cosine_greedy_kernel,\
    cpu_parallel_cosine_greedy_kernel


class CPUParallelCosineGreedy(BaseSimilarity):
    score_datatype = [("score", np.float32), ("matches", np.int32)]
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

        self.kernel = cpu_parallel_cosine_greedy_kernel(
            tolerance=self.tolerance,
            shift=self.shift,
            mz_power=self.mz_power,
            int_power=self.int_power,
            match_limit=self.match_limit,
            batch_size=self.batch_size,
        )

    def _matrix_with_numpy_output(
        self,
        references: List[Spectrum],
        queries: List[Spectrum],
        is_symmetric: bool = False,
    ) -> np.ndarray:
        if is_symmetric:
            warnings.warn("is_symmetric is ignored here, it has no effect.")

        refs = tuple(r.peaks.to_numpy for r in references)
        ques = tuple(q.peaks.to_numpy for q in queries)

        result = self.kernel(
            refs,
            ques
        )
            
        return np.rec.fromarrays(
            result,
            dtype=self.score_datatype,
        )

    def pair(self, reference: Spectrum, query: Spectrum) -> float:
        result_mat = self.matrix([reference], [query])
        return np.asarray(result_mat.squeeze(), dtype=self.score_datatype)
    
    def matrix(
        self,
        references: List[Spectrum],
        queries: List[Spectrum],
        array_type: Literal["numpy"] = "numpy",
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
        else:
            raise NotImplementedError()