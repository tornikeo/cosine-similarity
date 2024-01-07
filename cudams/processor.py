from pydantic import BaseModel, Field
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

from typing import Tuple, List, Literal
import numpy as np
from matchms.typing import SpectrumType
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.similarity import CosineGreedy
from cudams.cosine import similarity

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

class CosineGreedyResults:
    def __init__(
        self, config: Config, files: list[Path], len_references: int, len_queries: int
    ) -> None:
        self.files = files
        self.len_references = len_references
        self.len_queries = len_queries
        self.config = config
        assert all([f.exists() for f in self.files])

    def __str__(self) -> str:
        return f"CosineGreedyResults {str(self.config)}"

    def to_full_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        BATCH_SIZE = self.config.BATCH_SIZE
        R = math.ceil(self.len_references / BATCH_SIZE) * BATCH_SIZE
        Q = math.ceil(self.len_queries / BATCH_SIZE) * BATCH_SIZE

        G = np.empty((R, Q, 2), dtype="float32")
        Ov = np.empty((R, Q, 1), dtype="uint8")

        files = sorted([f for f in self.files])
        for file in files:
            rstart, rend, qstart, qend = name2idx(file)
            if "score" in file.stem:
                G[rstart:rend, qstart:qend] = np.load(file)
            elif "ovfl" in file.stem:
                Ov[rstart:rend, qstart:qend] = np.load(file)
        return G, Ov

    def to_pandas(self) -> pd.DataFrame:
        results = pd.DataFrame(
            [], columns=["Reference", "Query", "Score", "Num_Matches"]
        )
        # scores = sorted(gpu_output_dir.glob('*.score.npy'))
        scores = [f for f in self.files if f.stem.endswith("score")]
        for score in scores:
            rstart, rend, qstart, qend = name2idx(score)
            score = np.load(score)
            # Condition query
            pairs_relative = np.argwhere(score[..., 0] >= self.min_score)
            # We have to pad pairs with their actual locations on full grid
            pairs_absolute = pairs_relative + [rstart, qstart]

            # score, num_matches = get_one_specific(ref_idx, que_idx)
            r, q = pairs_relative.T
            score, num_match = score[r, q].T

            r, q = pairs_absolute.T
            result = pd.DataFrame(
                dict(
                    Reference=r.astype("uint32"),
                    Query=q.astype("uint32"),
                    Score=score.astype("float32"),
                    Num_Matches=num_match.astype("uint16"),
                )
            ).convert_dtypes()
            results = pd.concat([results, result], axis=0, copy=False)
        return results

    def filter(self, min_score):
        total_gbs = self.len_references * self.len_queries * 4 * 2 * 1e9
        self.min_score = min_score
        if min_score < 0.5 and total_gbs > 1:
            warnings.warn(f"min_score <= .5 on large results requres a LOT of RAM.")
        return self


def file_saver_worker(
    gpu_output_dir: Path,
    batch_size: int,
    name1: str,
    name2: str,
    rstart: int,
    rend: int,
    qstart: int,
    qend: int,
) -> None:
    BATCH_SIZE = batch_size
    ex_shm = shared_memory.SharedMemory(name=name1)
    out = np.ndarray(
        shape=(BATCH_SIZE, BATCH_SIZE, 2),
        dtype=np.float32,
        buffer=ex_shm.buf,
    )
    np.save(
        gpu_output_dir / f"{rstart}-{rend}.{qstart}-{qend}.score.npy",
        out,
    )
    ex_shm.unlink()

    ex_shm = shared_memory.SharedMemory(name=name2)
    overflow = np.ndarray(
        shape=(BATCH_SIZE, BATCH_SIZE, 1),
        dtype=np.uint8,
        buffer=ex_shm.buf,
    )
    np.save(
        gpu_output_dir / f"{rstart}-{rend}.{qstart}-{qend}.ovfl.npy",
        overflow,
    )
    ex_shm.unlink()


def numpy_accumulator_worker(
    gpu_output_dir: Path,
    batch_size: int,
    name1: str,
    name2: str,
    rstart: int,
    rend: int,
    qstart: int,
    qend: int,
) -> None:
    pass


class CudaCosineGreedy(BaseSimilarity):
    score_datatype = np.float64

    def __init__(
        self,
        tolerance: float = 0.1,
        mz_power: float = 0.0,
        intensity_power: float = 1.0,
        shift: float = 0,
        batch_size: int = 1024,
        match_limit: int = 256,
    ):
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.int_power = intensity_power
        self.shift = shift
        self.batch_size = batch_size
        self.match_limit = match_limit

        self.kernel = None
        self.results = None
        if not cuda.is_available():
            warnings.warn(f"{self.__class__}: CUDA device seems unavailable.")

    def compile(self):
        self.kernel = compile(
            tolerance=self.tolerance,
            shift=self.shift,
            mz_power=self.mz_power,
            int_power=self.int_power,
            match_limit=self.match_limit,
            batch_size=self.batch_size,
        )

    def __str__(self) -> str:
        return self.config.model_dump_json(indent=1)

    def _matrix_with_numpy_output(
        self,
        references: List[SpectrumType],
        queries: List[SpectrumType],
        is_symmetric: bool = False,
    ) -> (np.ndarray, np.ndarray):
        BATCH_SIZE = self.batch_size
        dtype = self.score_datatype

        batches_r = []
        for bstart, bend in tqdm(
            argbatch(references, BATCH_SIZE), desc="Batch all references"
        ):
            rbatch = references[bstart:bend]
            rspec, rlen = spectra_peaks_to_tensor(rbatch, dtype=dtype)
            batches_r.append([rspec, rlen, bstart, bend])

        batches_q = []
        for bstart, bend in tqdm(
            argbatch(queries, BATCH_SIZE), desc="Batch all queries"
        ):
            qbatch = queries[bstart:bend]
            qspec, qlen = spectra_peaks_to_tensor(qbatch, dtype=dtype)
            batches_q.append([qspec, qlen, bstart, bend])

        batches_rq = list(product(batches_r, batches_q))

        TOTAL_BATCHES = len(batches_rq)

        R = math.ceil(len(references) / BATCH_SIZE) * BATCH_SIZE
        Q = math.ceil(len(queries) / BATCH_SIZE) * BATCH_SIZE
        
        result_output = np.empty((R, Q, 2), dtype="float32")
        result_overflow = np.empty((R, Q, 1), dtype="uint8")

        # We loop over all batchs in sequence
        for batch_i in tqdm(range(TOTAL_BATCHES)):
            # Each batch has own CUDA stream so that the GPU is as busy as possible

            out = np.empty(
                shape=(BATCH_SIZE, BATCH_SIZE, 2),
                dtype="float32",
            )

            overflow = np.empty(
                shape=(BATCH_SIZE, BATCH_SIZE, 1),
                dtype="uint8",
            )

            # We get our batch and lengths (lengths are different for different spectra)
            (rspec, rlen, rstart, rend), (qspec, qlen, qstart, qend) = batches_rq[
                batch_i
            ]
            lens = np.zeros((2, BATCH_SIZE), "int32")
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
                    (BATCH_SIZE, BATCH_SIZE, 2), dtype="float32"
                )
                overflow_cu = cuda.device_array(
                    (BATCH_SIZE, BATCH_SIZE, 1), dtype="uint8"
                )

                # We order the stream to copy input data to GPU RAM
                rspec_cu = cuda.to_device(rspec)
                qspec_cu = cuda.to_device(qspec)
                lens_cu = cuda.to_device(lens)

                # We order the stream to execute kernel (this is scheduled, it will execute, but we can't force it)
                self.kernel(
                    rspec_cu, qspec_cu, lens_cu, out_cu, overflow_cu
                )

                # We order a data return
                out_cu.copy_to_host(out)
                overflow_cu.copy_to_host(overflow)

                result_output[rstart:rend, qstart:qend] = out
                result_overflow[rstart:rend, qstart:qend] = overflow
                
                # We wait for all streams to finish their work everywhere
                cuda.synchronize()

        return result_output, result_overflow

    def matrix(
        self,
        references: List[SpectrumType],
        queries: List[SpectrumType],
        array_type: Literal["numpy", "sparse", "memmap"] = "numpy",
        is_symmetric: bool = False,
    ) -> np.ndarray | CosineGreedyResults:
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
        """
        assert self.kernel is not None, "Kernel isn't compiled - use .compile() first"
        
        if array_type == "numpy":
            return self._matrix_with_numpy_output(references, queries, is_symmetric=is_symmetric)
        
        BATCH_SIZE = self.batch_size
        dtype = self.score_datatype

        batches_r = []
        for bstart, bend in tqdm(
            argbatch(references, BATCH_SIZE), desc="Batch all references"
        ):
            rbatch = references[bstart:bend]
            rspec, rlen = spectra_peaks_to_tensor(rbatch, dtype=dtype)
            batches_r.append([rspec, rlen, bstart, bend])

        batches_q = []
        for bstart, bend in tqdm(
            argbatch(queries, BATCH_SIZE), desc="Batch all queries"
        ):
            qbatch = queries[bstart:bend]
            qspec, qlen = spectra_peaks_to_tensor(qbatch, dtype=dtype)
            batches_q.append([qspec, qlen, bstart, bend])

        batches_rq = list(product(batches_r, batches_q))
        streams = [cuda.stream() for _ in range(len(batches_rq))]

        TOTAL_BATCHES = len(batches_rq)

        output_dir = mkdir(self.output_dir, clean=True)

        R = math.ceil(len(references) / BATCH_SIZE) * BATCH_SIZE
        Q = math.ceil(len(queries) / BATCH_SIZE) * BATCH_SIZE
        
        output_arr = np.empty((R, Q, 2), dtype="float32")

        # We initialize a pool of 3 workers that will offload results to disk
        with ThreadPool(3) as pool:
            # We loop over all batchs in sequence
            for batch_i in tqdm(range(TOTAL_BATCHES)):
                # Each batch has own CUDA stream so that the GPU is as busy as possible
                stream = streams[batch_i]

                # Shared memory allows pool workers to read array without copying it
                out_shm = shared_memory.SharedMemory(
                    create=True, size=(BATCH_SIZE * BATCH_SIZE * 2 * 4)
                )
                out = np.ndarray(
                    shape=(BATCH_SIZE, BATCH_SIZE, 2),
                    dtype="float32",
                    buffer=out_shm.buf,
                )

                overflow_shm = shared_memory.SharedMemory(
                    create=True, size=(BATCH_SIZE * BATCH_SIZE * 1 * 1)
                )
                overflow = np.ndarray(
                    shape=(BATCH_SIZE, BATCH_SIZE, 1),
                    dtype="uint8",
                    buffer=overflow_shm.buf,
                )

                # We get our batch and lengths (lengths are different for different spectra)
                (rspec, rlen, rstart, rend), (qspec, qlen, qstart, qend) = batches_rq[
                    batch_i
                ]
                lens = np.zeros((2, BATCH_SIZE), "int32")
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
                        (BATCH_SIZE, BATCH_SIZE, 2), dtype="float32", stream=stream
                    )
                    overflow_cu = cuda.device_array(
                        (BATCH_SIZE, BATCH_SIZE, 1), dtype="uint8", stream=stream
                    )

                    # We order the stream to copy input data to GPU RAM
                    rspec_cu = cuda.to_device(rspec, stream=stream)
                    qspec_cu = cuda.to_device(qspec, stream=stream)
                    lens_cu = cuda.to_device(lens, stream=stream)

                    # We order the stream to execute kernel (this is scheduled, it will execute, but we can't force it)
                    self.kernel(
                        rspec_cu, qspec_cu, lens_cu, out_cu, overflow_cu, stream=stream
                    )

                    # We order a data return
                    out_cu.copy_to_host(out, stream=stream)
                    overflow_cu.copy_to_host(overflow, stream=stream)

                    # We create a function that will execute when this stream is done working
                    # It is important to be quick here - so main work of writing to disk
                    # Is handled by pool workers, not callback stream.
                    def end_of_stream_callback(stream, status, args):
                        pool.apply_async(
                            file_saver_worker,
                            args=args,
                            error_callback=lambda e: print("Thread error", e),
                        )

                    stream.add_callback(
                        callback=end_of_stream_callback,
                        arg=[
                            output_dir,
                            BATCH_SIZE,
                            out_shm.name,
                            overflow_shm.name,
                            rstart,
                            rend,
                            qstart,
                            qend,
                        ],
                    )
        pool.join()
        # We wait for all streams to finish their work everywhere
        cuda.synchronize()

        files = sorted(list(output_dir.glob("*.npy")))
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


class CpuCosineGreedy(BaseSimilarity):
    score_datatype = np.float64
    def __init__(
        self,
        tolerance: float = 0.1,
        mz_power: float = 0.0,
        intensity_power: float = 1.0,
        shift: float = 0,
        batch_size: int = 1024
    ):
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.int_power = intensity_power
        self.shift = shift
        self.batch_size = batch_size

        self.kernel = None
        self.results = None
        if not cuda.is_available():
            warnings.warn(f"{self.__class__}: CUDA device seems unavailable.")

    def matrix(self, 
               references: List[SpectrumType], 
               queries: List[SpectrumType], 
               array_type: Literal['numpy'] = "numpy", 
               is_symmetric: bool = False) -> np.ndarray:
        
        BATCH_SIZE = self.batch_size
        
        refs = list([r.peaks.to_numpy for r in references])
        ques = list([q.peaks.to_numpy for q in queries])

        rlims = argbatch(refs, BATCH_SIZE)
        qlims = argbatch(ques, BATCH_SIZE)
        
        batches_rq = list(product(rlims, qlims))

        R = math.ceil(len(references) / BATCH_SIZE) * BATCH_SIZE
        Q = math.ceil(len(queries) / BATCH_SIZE) * BATCH_SIZE
        
        result_output = np.empty((R, Q, 2), dtype="float32")

        for (rstart, rend), (qstart, qend) in tqdm(batches_rq, total=len(batches_rq)):
            rspec = refs[rstart:rend]
            qspec = ques[qstart:qend]
            out_true = np.full((BATCH_SIZE, BATCH_SIZE, 2), fill_value=0, dtype='float32')
            for (i, spec1), (j, spec2) in product(enumerate(rspec), enumerate(qspec)):
                    score = similarity(
                        spec1,
                        spec2,
                        tolerance=self.tolerance,
                        shift=self.shift,
                        mz_power=self.mz_power,
                        int_power=self.int_power,
                    )
                    if score is not None:
                        out_true[i,j,0] = score[0]
                        out_true[i,j,1] = score[1]
            # np.save(cpu_output_dir / f'{rstart}-{rend}.{qstart}-{qend}.score.npy', out_true)
            result_output[rstart:rend, qstart:qend] = out_true
        return result_output