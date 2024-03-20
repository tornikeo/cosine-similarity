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
    max_spectra_length: int = 2048,
    batch_size: int = 2048,
    is_symmetric: bool = False,
) -> callable:
    if is_symmetric:
        import warnings

        warnings.warn("no effect from is_symmetric, it is not yet implemented")

    MAX_SPECTRA_LENGTH = max_spectra_length
    MATCH_LIMIT = match_limit
    R, Q = batch_size, batch_size
    # Since the first outer loop in acc step is iterating over references,
    # It is faster to have a block that runs same reference, over multiple
    # queries, so that thread divergence is minimized
    THREADS_PER_BLOCK_X = 1
    THREADS_PER_BLOCK_Y = 512
    THREADS_PER_BLOCK = (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)
    BLOCKS_PER_GRID_X = math.ceil(R / THREADS_PER_BLOCK_X)
    BLOCKS_PER_GRID_Y = math.ceil(Q / THREADS_PER_BLOCK_Y)
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

        thread_i = cuda.threadIdx.x
        thread_j = cuda.threadIdx.y
        block_size_x = cuda.blockDim.x
        block_size_y = cuda.blockDim.y

        # We aren't out of the RxQ grid
        if not( i < R and j < Q ):
            return

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

        # matches = cuda.local.array((2, MATCH_LIMIT), types.int16)

        # overflow = 0
        # diff = 0.0
        # peak1_idx = 0
        # peak2_idx = 0
        # while True:
        # # for peak1_idx in range(rleni):
        # #     if overflow > 0:
        # #         break
        #     mz = spec1_mz[peak1_idx]
        #     mz2 = spec2_mz[peak2_idx] + shift
        # #     for peak2_idx in range(qlenj)
        #     # peak1_idx = peak k
        #     # peak2_idx = max(min(lowest_idx, peak2_idx), qlenj - 1)
        #     diff = mz2 - mz
        #     # if abs(diff) <= tolerance and num_match < MATCH_LIMIT:
        #     matches[0, num_match] = peak1_idx
        #     matches[1, num_match] = peak2_idx

        #     if abs(diff) <= tolerance:
        #         num_match += 1

        #     num_match = min(num_match, MATCH_LIMIT - 1)

        #     if num_match >= MATCH_LIMIT - 1:
        #         overflow = 1
        #         break

        #     if diff < tolerance:
        #         lowest_idx = peak2_idx + 1

        #     # if diff > tolerance:
        #     #     peak1_idx += 1

        #     peak2_idx += 1

        #     if peak2_idx >= qlenj:
        #         peak1_idx += 1
        #         peak2_idx = lowest_idx

        #     if peak1_idx >= rleni:
        #         break

        # out[2, i, j] = overflow
        # overflow = types.boolean(False)

        ## With  blocksize 1x256, we know R is going to be same for all threads within same block
        # spec1_mz_sh = cuda.shared.array(MAX_SPECTRA_LENGTH, types.float32)
        # spec1_int_sh = cuda.shared.array(MAX_SPECTRA_LENGTH, types.float32)
        spec1_sh = cuda.shared.array((2, MAX_SPECTRA_LENGTH), types.float32)
        # spec1_mz_sh = cuda.shared.array(MAX_SPECTRA_LENGTH, types.float32)
        # spec1_int_sh = cuda.shared.array(MAX_SPECTRA_LENGTH, types.float32)

        score_norm_spec1_sh = cuda.shared.array(1, types.float32)
        score_norm_spec1_sh[0] = 0
        cuda.syncthreads()

        scale = (MAX_SPECTRA_LENGTH + cuda.blockDim.y - 1) // cuda.blockDim.y
        norm_accum = 0.0
        for offset in range(0, scale):
            where = thread_j * scale + offset
            if where < MAX_SPECTRA_LENGTH and where < rleni:
                mz_ = spec1_mz[where]
                int_ = spec1_int[where]
                norm_accum += (mz_ ** mz_power * int_ ** int_power) ** 2
                # cuda.atomic.add(score_norm_spec1_sh, 0, (mz_ ** mz_power * int_ ** int_power) ** 2)

                spec1_sh[0, where] = mz_
                spec1_sh[1, where] = int_

        matches = cuda.local.array((2, MATCH_LIMIT), types.uint16)

        spec1_mz_sh = spec1_sh[0]
        spec1_int_sh = spec1_sh[1]

        ## Needs blocksize small enough, i.e. 1,4?
        # matches_all = cuda.shared.array((THREADS_PER_BLOCK_Y, 2, MATCH_LIMIT), types.int16)
        # matches = matches_all[thread_j]
        cuda.atomic.add(score_norm_spec1_sh, 0, norm_accum)
        cuda.syncthreads()
        score_norm_spec1 = score_norm_spec1_sh[0]

        ovf = 0
        for peak1_idx in range(rleni):
            if ovf == 1:
                break
            mz = spec1_mz_sh[peak1_idx]
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
                        ovf = 1  # This is the errorcode for overflow
                        # overflow = True
                        break

        out[2, i, j] = ovf
        # out[2,i,j] = overflow
        ## Debug checkpoint
        # out[i, j, 0] = 1
        # out[i, j, 1] = num_match
        # return
        if num_match == 0:
            return

        score_norm = types.float32(1.0)
        # score_norm_spec1 = types.float32(0.0)
        score_norm_spec2 = types.float32(0.0)

        # TODO: It is faster to pre-compute this for every R,Q beforehand and pass it in.
        # score_norm_spec1 = 0
        # for peak1_idx in range(rleni):
        #     score_norm_spec1 += (
        #         (spec1_mz_sh[peak1_idx] ** mz_power)
        #         * (spec1_int_sh[peak1_idx] ** int_power)
        #     ) ** 2

        for peak2_idx in range(qlenj):
            score_norm_spec2 += (
                (spec2_mz[peak2_idx] ** mz_power)
                * (spec2_int[peak2_idx] ** int_power)
            ) ** 2
        score_norm = math.sqrt(score_norm_spec1) * math.sqrt(score_norm_spec2)

        # Debug checkpoint
        # out[i, j, 0] = score_norm
        # out[i, j, 1] = num_match
        # return

        # TODO: VERY slow - Bubble sort (This should also be done in several threads)
        # We need two cases, bubble sort up to 50 elems is fine
        score = types.float32(0.0)
        used_matches = types.uint16(0)
        for _ in range(0, num_match):
            max_prod = types.float32(-1.0)
            max_peak1_idx = types.uint16(0xFFFF)
            max_peak2_idx = types.uint16(0xFFFF)

            for sj in range(0, num_match):
                if matches[0, sj] != 0xFFFF:
                    peak1_idx = matches[0, sj]
                    peak2_idx = matches[1, sj]

                    power_prod_spec1 = (spec1_mz_sh[peak1_idx] ** mz_power) * (
                        spec1_int_sh[peak1_idx] ** int_power
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
                    if (matches[0, sj] == max_peak1_idx or matches[1, sj] == max_peak2_idx):
                        matches[0, sj] = 0xFFFF # "Remove" it
                        matches[1, sj] = 0xFFFF # "Remove" it
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
    """
    Work-in-progress CPU-Parallel implementation of Cosine Greedy. With CUDA cosine greedy (even on colab), this function
    isn't worth using. Mostly used for benchmarking and comparison.
    """
    @numba.jit(nopython=True, parallel=True)
    def cpu_kernel(
        refs: list[np.ndarray],
        ques: list[np.ndarray],
    ) -> np.ndarray:
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
