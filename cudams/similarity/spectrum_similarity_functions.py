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
        ## Get global indices
        i, j = cuda.grid(2)
        thread_i = cuda.threadIdx.x
        thread_j = cuda.threadIdx.y
        block_size_x = cuda.blockDim.x
        block_size_y = cuda.blockDim.y

        # We aren't out of the RxQ grid
        if i >= R or j >= Q:
            return

        # Init values (we expect these to be uninitialized)
        out[0, i, j] = 0
        out[1, i, j] = 0
        out[2, i, j] = 0

        # Unpack mz and int from arrays
        rmz = rspec[0] 
        rint = rspec[1]
        qmz = qspec[0]
        qint = qspec[1]

        # Get actual length of R and Q spectra
        rleni = lens[0, i]
        qlenj = lens[1, j]

        # Check that both spectra are non-empty
        if rleni == 0 or qlenj == 0:
            return

        spec1_mz = rmz[i]
        spec1_int = rint[i]

        spec2_mz = qmz[j]
        spec2_int = qint[j]

        lowest_idx = types.int32(0)
        num_match = types.int32(0)

        ## PART 1, CALCULATE SCORE NORMS
        # Since blocksize is 1xN, with RxQ layout, we know that all threads in a block
        # work on the same R. So, we calculate R-norm together, and store it in shared memory so all threads can access it later on
        score_norm_spec1_sh = cuda.shared.array(1, types.float32)
        score_norm_spec1_sh[0] = 0
        cuda.syncthreads()

        scale = (rleni + cuda.blockDim.y - 1) // cuda.blockDim.y
        norm_accum = 0.0
        for ix in range(thread_j * scale, min((thread_j + 1) * scale, rleni)):
            norm_accum += (spec1_mz[ix] ** mz_power * spec1_int[ix] ** int_power) ** 2

        # spec1_mz_sh = spec1_sh[0]
        # spec1_int_sh = spec1_sh[1]
        spec1_mz_sh = spec1_mz
        spec1_int_sh = spec1_int

        ## Needs blocksize small enough, i.e. 1,4?
        # matches_all = cuda.shared.array((THREADS_PER_BLOCK_Y, 2, MATCH_LIMIT), types.int16)
        # matches = matches_all[thread_j]
        cuda.atomic.add(score_norm_spec1_sh, 0, norm_accum)
        cuda.syncthreads()
        score_norm_spec1 = score_norm_spec1_sh[0]
        
        score_norm_spec2 = types.float32(0.0)
        for ix in range(qlenj):
            score_norm_spec2 += (spec2_mz[ix] ** mz_power * spec2_int[ix] ** int_power) ** 2
        score_norm = math.sqrt(score_norm_spec1) * math.sqrt(score_norm_spec2)
        
        matches = cuda.local.array(MATCH_LIMIT, types.int32)
        values = cuda.local.array(MATCH_LIMIT, types.float32)

        ovf = False
        for peak1_idx in range(rleni):
            if ovf:
                break
            mz_r = spec1_mz_sh[peak1_idx]
            int_r = spec1_int_sh[peak1_idx]
            for peak2_idx in range(lowest_idx, qlenj):
                mz_q = spec2_mz[peak2_idx] + shift
                delta = mz_q - mz_r

                if delta > tolerance:
                    break
                if delta < -tolerance:
                    lowest_idx = peak2_idx + 1

                else:
                    if not ovf:
                        int_q = spec2_int[peak2_idx]
                        matches[num_match] = (peak1_idx << 16) | peak2_idx
                        power_prod_spec1 = mz_r ** mz_power * int_r ** int_power
                        power_prod_spec2 = mz_q ** mz_power * int_q ** int_power
                        prod = power_prod_spec1 * power_prod_spec2
                        values[num_match] = prod
                        num_match += 1
                        ovf = num_match >= MATCH_LIMIT  # This is the errorcode for overflow

        out[2, i, j] = ovf
        # out[2,i,j] = overflow
        ## Debug checkpoint
        # out[i, j, 0] = 1
        # out[i, j, 1] = num_match
        # return
        if num_match == 0:
            return
        
        # score_norm = types.float32(1.0)
        # score_norm_spec1 = types.float32(0.0)

        # TODO: It is faster to pre-compute this for every R,Q beforehand and pass it in.
        # score_norm_spec1 = 0
        # for peak1_idx in range(rleni):
        #     score_norm_spec1 += (
        #         (spec1_mz_sh[peak1_idx] ** mz_power)
        #         * (spec1_int_sh[peak1_idx] ** int_power)
        #     ) ** 2
        
        # score_norm_spec2 = types.float32(0.0)
        # for peak2_idx in range(qlenj):
        #     score_norm_spec2 += (
        #         (spec2_mz[peak2_idx] ** mz_power)
        #         * (spec2_int[peak2_idx] ** int_power)
        #     ) ** 2
        # score_norm = math.sqrt(score_norm_spec1) * math.sqrt(score_norm_spec2)

        # Debug checkpoint
        # out[i, j, 0] = score_norm
        # out[i, j, 1] = num_match
        # return

        ## Non-recursive merge-sort
        
        # temp_matches = cuda.local.array((2, MATCH_LIMIT), types.uint16)
        # temp_values = cuda.local.array(MATCH_LIMIT, types.float32)

        
        # # k = types.uint16(1)
        # # used_matches += 1
        # k = 1
        # while k < num_match:
        #     for left in range(0, num_match - k, k * 2):
        #         rght = left + k
        #         rend = rght + k
                
        #         if rend > num_match:
        #             rend = num_match
        #         m = left; i = left; j = rght;
        #         while i < rght and j < rend:
        #             if values[i] <= values[j]:
        #                 temp_matches[0, m] = matches[0, i]; 
        #                 temp_matches[1, m] = matches[1, i]; 
        #                 temp_values[m] = values[i]; 
        #                 i+=1
        #             else:
        #                 temp_matches[0, m] = matches[0, j]; 
        #                 temp_matches[1, m] = matches[1, j]; 
        #                 temp_values[m] = values[j]; 
        #                 j+=1
        #             m+=1
                
        #         while i < rght:
        #             temp_matches[0, m] = matches[0, i]; 
        #             temp_matches[1, m] = matches[1, i]; 
        #             temp_values[m] = values[i]; 
        #             i+=1; m+=1;
                
        #         while j < rend:
        #             temp_matches[0, m] = matches[0, j]; 
        #             temp_matches[1, m] = matches[1, j]; 
        #             temp_values[m] = values[j]; 
        #             j+=1; m+=1;
                
        #         for m in range(left, rend):
        #             matches[0, m] = temp_matches[0, m]; 
        #             matches[1, m] = temp_matches[1, m]; 
        #             values[m] = temp_values[m]; 
        #     k *= 2
        
        # # out[1, i, j] = matches[0, 0]
        # used_r = cuda.local.array(MAX_SPECTRA_LENGTH, types.boolean)
        # used_q = cuda.local.array(MAX_SPECTRA_LENGTH, types.boolean)

        # for m in range(0, MAX_SPECTRA_LENGTH):
        #     used_r[m] = False
        #     used_q[m] = False

        # used_matches = 0
        # score = 0.0
        # for m in range(0, num_match):
        #     j = (num_match - 1) - m
        #     peak1_idx = matches[0, j]; 
        #     peak2_idx = matches[1, j]; 
        #     if (not used_r[peak1_idx]) and (not used_q[peak2_idx]):
        #         used_r[peak1_idx] = True
        #         used_q[peak2_idx] = True
        #         score += values[j];
        #         used_matches += 1

        # TODO: VERY slow - Bubble sort (This should also be done in several threads)
        # We need two cases, bubble sort up to 50 elems is fine
        score = types.float32(0.0)
        used_matches = types.uint16(0)
        for _ in range(0, num_match):
            max_prod = types.float32(-1.0)
            max_peak1_idx = 0
            max_peak2_idx = 0

            for sj in range(0, num_match):
                c = matches[sj]
                if c != -1:
                    peak1_idx = c >> 16
                    peak2_idx = c & 0x0000_FFFF

                    # power_prod_spec1 = (spec1_mz_sh[peak1_idx] ** mz_power) * (
                    #     spec1_int_sh[peak1_idx] ** int_power
                    # )
                    # power_prod_spec2 = (spec2_mz[peak2_idx] ** mz_power) * (
                    #     spec2_int[peak2_idx] ** int_power
                    # )
                    # prod = power_prod_spec1 * power_prod_spec2
                    prod = values[sj]

                    # > was changed to >= and that took 2 weeks... also finding that 'mergesort' in original similarity algorithm
                    # is what can prevent instability.
                    if prod >= max_prod:
                        max_prod = prod
                        max_peak1_idx = peak1_idx
                        max_peak2_idx = peak2_idx

            if max_prod != -1:
                for sj in range(0, num_match):
                    c = matches[sj]
                    if c != -1:
                        peak1_idx = c >> 16
                        peak2_idx = c & 0x0000_FFFF
                        if (peak1_idx == max_peak1_idx or peak2_idx == max_peak2_idx):
                            matches[sj] = -1 # "Remove" it

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

        out[0, i, j] = score / score_norm
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
