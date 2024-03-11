import math

import numba
import numpy as np
import torch
from matchms.similarity.spectrum_similarity_functions import (
    collect_peak_pairs, score_best_matches)
from numba import cuda, pndindex, prange, types
from torch import Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"


def cosine_greedy_kernel(
    tolerance: float = 0.1,
    shift: float = 0,
    mz_power: float = 0.0,
    int_power: float = 1.0,
    match_limit: int = 1024,
    batch_size: int = 2048,
    is_symmetric: bool = False,
) -> callable:
    if is_symmetric:
        import warnings

        warnings.warn("no effect from is_symmetric, it is not yet implemented")

    MATCH_LIMIT = match_limit
    R, Q = batch_size, batch_size
    # Since the first outer loop in acc step is iterating over references,
    # It is faster to have a block that runs same reference, over multiple
    # queries, so that thread divergence is minimzed
    THREADS_PER_BLOCK = (1, 512 + 256)
    BLOCKS_PER_GRID_X = math.ceil(R / THREADS_PER_BLOCK[0])
    BLOCKS_PER_GRID_Y = math.ceil(Q / THREADS_PER_BLOCK[1])
    BLOCKS_PER_GRID = (BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y)

    @cuda.jit
    def _kernel(
        rspec,
        qspec,
        lens,
        out,
    ):
        i, j = cuda.grid(2)
        # score_out = out[0]
        # matches_out = out[1]
        # overflow_out = out[2]

        # thread_i = cuda.threadIdx.x
        # thread_j = cuda.threadIdx.y
        # block_size_x = cuda.blockDim.x
        # block_size_y = cuda.blockDim.y

        # We aren't out of the RxQ grid
        if i < R and j < Q:
            # Init values (we expect these to be uninitialized)
            out[0, i, j] = 0
            out[1, i, j] = 0
            out[2, i, j] = 0

            # mem = cuda.shared.array((4, 4, 4, 32), types.float32)
            rmz = rspec[0]  # rspec shape is [2, 2048, 256], rmz shape is [2048, 256]
            rint = rspec[1]
            qmz = qspec[0]
            qint = qspec[1]
            # In this i,j, We get length of r and q spectrums
            # since they are batched, there might be extra filler elements
            rlen = lens[0]
            qlen = lens[1]

            rleni = rlen[i]
            qlenj = qlen[j]

            # When we have batch that is incomplete (size is indivisible by B)
            # we return quickly to avoid writing garbage there.
            if rleni == 0 or qlenj == 0:
                return

            spec1_mz = rmz[i]
            spec1_int = rint[i]

            spec2_mz = qmz[j]
            spec2_int = qint[j]

            lowest_idx = types.int32(0)
            num_match = types.int32(0)

            matches = cuda.local.array((2, MATCH_LIMIT), types.int16)
            for peak1_idx in range(rleni):
                mz = spec1_mz[peak1_idx]
                low_bound = mz - tolerance
                high_bound = mz + tolerance

                for peak2_idx in range(lowest_idx, qlenj):
                    mz2 = spec2_mz[peak2_idx] + shift
                    if mz2 > high_bound:
                        break
                    if mz2 < low_bound:
                        lowest_idx = peak2_idx + 1
                    else:
                        if num_match < MATCH_LIMIT:
                            matches[0, num_match] = peak1_idx
                            matches[1, num_match] = peak2_idx
                            num_match += 1
                        else:
                            out[2, i, j] = 1  # This is the errorcode for overflow
                            break
            ## Debug checkpoint
            # out[i, j, 0] = 1
            # out[i, j, 1] = num_match
            # return

            if num_match == 0:
                return

            score_norm = types.float64(1.0)
            score_norm_spec1 = types.float64(0.0)
            score_norm_spec2 = types.float64(0.0)

            # TODO: It is faster to pre-compute this for every R,Q beforehand and pass it in.
            for peak1_idx in range(rleni):
                score_norm_spec1 += (
                    (spec1_mz[peak1_idx] ** mz_power)
                    * (spec1_int[peak1_idx] ** int_power)
                ) ** 2
            for peak2_idx in range(qlenj):
                score_norm_spec2 += (
                    (spec2_mz[peak2_idx] ** mz_power)
                    * (spec2_int[peak2_idx] ** int_power)
                ) ** 2
            score_norm = math.sqrt(score_norm_spec1 * score_norm_spec2)

            # Debug checkpoint
            # out[i, j, 0] = score_norm
            # out[i, j, 1] = num_match
            # return

            # TODO: VERY slow - Bubble sort (This should also be done in several threads)
            # We need two cases, bubble sort up to 50 elems is fine
            score = types.float32(0.0)
            used_matches = types.int32(0.0)
            for _ in range(0, num_match):
                max_prod = types.float32(-1.0)
                max_peak1_idx = types.int32(-1)
                max_peak2_idx = types.int32(-1)

                for sj in range(0, num_match):
                    if matches[0, sj] != -1:
                        peak1_idx = matches[0, sj]
                        peak2_idx = matches[1, sj]

                        power_prod_spec1 = (spec1_mz[peak1_idx] ** mz_power) * (
                            spec1_int[peak1_idx] ** int_power
                        )
                        power_prod_spec2 = (spec2_mz[peak2_idx] ** mz_power) * (
                            spec2_int[peak2_idx] ** int_power
                        )
                        prod = power_prod_spec1 * power_prod_spec2

                        # > was changed to >= and that took 2 weeks... also finding that 'mergesort' in original similarity algorithm
                        # is what can prevent instability.
                        if prod >= max_prod:
                            max_prod = prod
                            max_peak1_idx = peak1_idx
                            max_peak2_idx = peak2_idx

                # Debug checkpoint
                # out[i, j, 0] = max_prod
                # out[i, j, 1] = used_matches
                # return

                if max_prod != -1:
                    for sj in range(0, num_match):
                        if (
                            matches[0, sj] == max_peak1_idx
                            or matches[1, sj] == max_peak2_idx
                        ):
                            matches[0, sj] = -1  # "Remove" it
                            matches[1, sj] = -1  # "Remove" it
                    score += max_prod
                    used_matches += 1
                else:
                    break

            # debug checkpoint
            # out[i, j, 0] = score_norm
            # out[i, j, 1] = used_matches
            # return

            # out[i, j, 0] = matches[0, MATCH_LIMIT-1]
            # out[i, j, 1] = matches[1, MATCH_LIMIT-1]

            # out[i, j, 0] = matches[0, 0]
            # out[i, j, 1] = matches[1, 0]
            # return

            score = score / score_norm
            out[0, i, j] = score
            out[1, i, j] = used_matches

    def kernel(
        rspec_cu,
        qspec_cu,
        lens_cu,
        out_cu,
        stream: cuda.stream = None,
    ):
        _kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK, stream](
            rspec_cu,
            qspec_cu,
            lens_cu,
            out_cu,
        )

    return kernel


def cpu_parallel_cosine_greedy_kernel(
    tolerance: float = 0.1,
    shift: float = 0,
    mz_power: float = 0.0,
    int_power: float = 1.0,
    match_limit: int = 1024,
    batch_size: int = 2048,
    is_symmetric: bool = False,
) -> callable:
    @numba.jit(nopython=True, parallel=True)
    def cpu_kernel(
        refs: list[np.ndarray],
        ques: list[np.ndarray],
    ) -> np.ndarray:
        """Returns matrix of cosine similarity scores between all-vs-all vectors of
        references and queries.

        Parameters
        ----------
        references
            Reference vectors as 2D numpy array. Expects that vector_i corresponds to
            references[i, :].
        queries
            Query vectors as 2D numpy array. Expects that vector_i corresponds to
            queries[i, :].

        Returns
        -------
        scores
            Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
            between the vectors references[i, :] and queries[j, :].
        """
        R = len(refs)
        Q = len(ques)

        scores = np.zeros((2, R, Q), dtype=np.float32)
        for i, j in pndindex(R, Q):
            ref = refs[i]
            que = ques[j]
            matching_pairs = collect_peak_pairs(ref,
                                                que,
                                                tolerance=tolerance,
                                                shift=shift,
                                                mz_power=mz_power,
                                                intensity_power=int_power)
            if matching_pairs is None:
                continue;
            matching_pairs = matching_pairs[np.argsort(matching_pairs[:, 2], kind='mergesort')[::-1], :]
            score, matches = score_best_matches(matching_pairs, que, ref, mz_power, int_power)
            scores[0, i, j] = score
            scores[1, i, j] = matches
        return scores

    return cpu_kernel
