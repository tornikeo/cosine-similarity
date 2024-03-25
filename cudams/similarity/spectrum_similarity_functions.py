import math

import numba
import numpy as np
import torch
from matchms.similarity.spectrum_similarity_functions import (
    collect_peak_pairs, score_best_matches)
from numba import cuda, pndindex, prange, types
from torch import Tensor


def cosine_greedy_kernel(
    tolerance: float = 0.1,
    shift: float = 0,
    mz_power: float = 0.0,
    int_power: float = 1.0,
    match_limit: int = 1024,
    batch_size: int = 2048,
    n_max_peaks: int = 2048,
    is_symmetric: bool = False,
) -> callable:
    if is_symmetric:
        import warnings
        warnings.warn("no effect from is_symmetric, it is not yet implemented")

    MATCH_LIMIT = match_limit
    N_MAX_PEAKS = n_max_peaks
    R, Q = batch_size, batch_size
    # Since the first outer loop in acc step is iterating over references,
    # It is faster to have a block that runs same reference, over multiple
    # queries, so that thread divergence is minimized
    THREADS_PER_BLOCK_X = 1
    THREADS_PER_BLOCK_Y = 512 + 256
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
        ## PREAMBLE:
        # Get global indices
        i, j = cuda.grid(2)
        thread_i = cuda.threadIdx.x
        thread_j = cuda.threadIdx.y
        block_size_x = cuda.blockDim.x
        block_size_y = cuda.blockDim.y
        
        # Check we aren't out of the max possible grid size
        if i >= R or j >= Q:
            return
        

        # Set zeros, since we know we are in the grid
        out[0, i, j] = 0
        out[1, i, j] = 0
        out[2, i, j] = 0

        # Get actual length of R and Q spectra
        rleni = lens[0, i]
        qlenj = lens[1, j]

        # We are in the grid, but current ij pair might be just a padding.
        # We check that and exit quickly if so.
        if rleni == 0 or qlenj == 0:
            return

        # We unpack mz and int from arrays
        rmz = rspec[0] 
        rint = rspec[1]
        qmz = qspec[0]
        qint = qspec[1]
        spec1_mz = rmz[i]
        spec1_int = rint[i]
        spec2_mz = qmz[j]
        spec2_int = qint[j]
        
        #### PART 1: CALCULATE SCORE NORMS ####
        # All the block shares the single R, so we calculate R's norm in a group
        # So, we calculate R-norm in a group, and store it in shared memory so all threads can access it later on
        score_norm_spec1_sh = cuda.shared.array(1, types.float32)
        score_norm_spec1_sh[0] = 0
        cuda.syncthreads()
        scale = max(8, (rleni + cuda.blockDim.y - 1) // cuda.blockDim.y)
        norm_accum = 0.0
        for ix in range(thread_j * scale, min((thread_j + 1) * scale, rleni)):
            norm_accum += (spec1_mz[ix] ** mz_power * spec1_int[ix] ** int_power) ** 2
        cuda.atomic.add(score_norm_spec1_sh, 0, norm_accum)
        cuda.syncthreads()
        score_norm_spec1 = score_norm_spec1_sh[0]
        
        # Since Qs are different, each thread calculates their own norm factor for Q
        score_norm_spec2 = types.float32(0.0)
        for ix in range(qlenj):
            score_norm_spec2 += (spec2_mz[ix] ** mz_power * spec2_int[ix] ** int_power) ** 2
        score_norm = math.sqrt(score_norm_spec1) * math.sqrt(score_norm_spec2)
        spec1_mz_sh = spec1_mz
        spec1_int_sh = spec1_int

        #### PART 2: Find matches ####
        # On GPU global memory, we allocate temporary arrays for values and matches
        matches = cuda.local.array(MATCH_LIMIT, types.int32)
        values = cuda.local.array(MATCH_LIMIT, types.float32)
        
        lowest_idx = types.int32(0)
        num_match = types.int32(0)
        
        overflow = False
        for peak1_idx in range(rleni):
            if overflow:
                break
            mz_r = spec1_mz_sh[peak1_idx]
            int_r = spec1_int_sh[peak1_idx]

            for peak2_idx in range(lowest_idx, qlenj):
                if overflow:
                    break
                mz_q = spec2_mz[peak2_idx]
                delta = mz_q + shift - mz_r
                if delta > tolerance:
                    break
                if delta < -tolerance:
                    lowest_idx = peak2_idx + 1
                else:
                    if not overflow:
                        int_q = spec2_int[peak2_idx]
                        power_prod_spec1 = mz_r ** mz_power * int_r ** int_power
                        power_prod_spec2 = mz_q ** mz_power * int_q ** int_power
                        prod = power_prod_spec1 * power_prod_spec2
                        # Binary trick! We pack two 16bit ints in 32bit int to use less memory
                        # since we know that largest imaginable peak index can fit in 13 bits
                        matches[num_match] = (peak1_idx << 16) | peak2_idx
                        values[num_match] = prod
                        num_match += 1
                        overflow = num_match >= MATCH_LIMIT  # This is the errorcode for overflow

        out[2, i, j] = overflow
        
        if num_match == 0:
            return

        # Debug checkpoint
        # out[i, j, 0] = score_norm
        # out[i, j, 1] = num_match
        # return

        #### PART: 2 ####
        # We use as non-recursive mergesort to order matches by the cosine produc
        # We need an O(MATCH_LIMIT) auxiliary memory for this.

        temp_matches = cuda.local.array(MATCH_LIMIT, types.int32)
        temp_values = cuda.local.array(MATCH_LIMIT, types.float32)

        k = types.int32(1)
        while k < num_match:
            for left in range(0, num_match - k, k * 2):
                rght = left + k
                rend = rght + k
                
                rend = min(rend, num_match)

                m = left; ix = left; jx = rght;
                while ix < rght and jx < rend:
                    mask = (values[ix] > values[jx])
                    temp_matches[m] = mask * matches[ix] + (1 - mask) * matches[jx]
                    temp_values[m] = mask * values[ix] + (1 - mask) * values[jx]
                    ix+=mask
                    jx+=(1-mask)
                    m+=1

                while ix < rght:
                    temp_matches[m] = matches[ix]; 
                    temp_values[m] = values[ix]; 
                    ix+=1; m+=1;
                
                while jx < rend:
                    temp_matches[m] = matches[jx]; 
                    temp_values[m] = values[jx]; 
                    jx+=1; m+=1;
                
                for m in range(left, rend):
                    matches[m] = temp_matches[m]; 
                    values[m] = temp_values[m]; 
            k *= 2


        #### PART: 3 Accumulate and deduplicate matches from largest to smallest ####
        used_r = cuda.local.array(N_MAX_PEAKS, types.boolean)
        used_q = cuda.local.array(N_MAX_PEAKS, types.boolean)

        for m in range(N_MAX_PEAKS):
            used_r[m] = False
            used_q[m] = False

        used_matches = 0
        score = 0.0
        for m in range(num_match):
            c = matches[m]
            peak1_idx = c >> 16
            peak2_idx = c & 0x0000_FFFF
            if (not used_r[peak1_idx]) and (not used_q[peak2_idx]):
                used_r[peak1_idx] = True
                used_q[peak2_idx] = True
                score += values[m];
                used_matches += 1
            

        #### PART 2: ALTENRNATIVE SORT+ACCUMULATE PATHWAY ####
        # This pathway is much faster when matches and average scores are *extremely* rare
        # TODO: We should compile both kernels, compare perfs and use fastest kernel.
        # else:
        # score = types.float32(0.0)
        # used_matches = types.uint16(0)
        # for _ in range(0, num_match):
        #     max_prod = types.float32(-1.0)
        #     max_peak1_idx = 0
        #     max_peak2_idx = 0

        #     for sj in range(0, num_match):
        #         c = matches[sj]
        #         if c != -1:
        #             peak1_idx = c >> 16
        #             peak2_idx = c & 0x0000_FFFF

        #             prod = values[sj]

        #             # > was changed to >= and that took 2 weeks... also finding that 'mergesort' in original similarity algorithm
        #             # is what can prevent instability.
        #             if prod >= max_prod:
        #                 max_prod = prod
        #                 max_peak1_idx = peak1_idx
        #                 max_peak2_idx = peak2_idx

        #     if max_prod != -1:
        #         for sj in range(0, num_match):
        #             c = matches[sj]
        #             if c != -1:
        #                 peak1_idx = c >> 16
        #                 peak2_idx = c & 0x0000_FFFF
        #                 if (peak1_idx == max_peak1_idx or peak2_idx == max_peak2_idx):
        #                     matches[sj] = -1 # "Remove" it

        #         score += max_prod
        #         used_matches += 1
        #     else:
        #         break

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
