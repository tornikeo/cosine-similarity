import math
import warnings
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
    """
    Compiles and returns a CUDA kernel function for calculating cosine similarity scores between spectra.

    Parameters:
    -----------
    tolerance : float, optional
        Tolerance parameter for m/z matching, by default 0.1.
    shift : float, optional
        Shift parameter for m/z matching, by default 0.
    mz_power : float, optional
        Power parameter for m/z intensity calculation, by default 0.
    int_power : float, optional
        Power parameter for intensity calculation, by default 1.
    match_limit : int, optional
        Maximum number of matches to consider, by default 1024.
    batch_size : int, optional
        Batch size for processing spectra, by default 2048.
    n_max_peaks : int, optional
        Maximum number of peaks to consider, by default 2048.
    is_symmetric : bool, optional
        Flag indicating if the similarity matrix is symmetric, by default False.

    Returns:
    --------
    callable
        CUDA kernel function for calculating cosine similarity scores.
    """

    if is_symmetric:
        warnings.warn("no effect from is_symmetric, it is not yet implemented")

    MATCH_LIMIT = match_limit
    N_MAX_PEAKS = n_max_peaks
    R, Q = batch_size, batch_size
    THREADS_PER_BLOCK_X = 1
    THREADS_PER_BLOCK_Y = 512
    THREADS_PER_BLOCK = (THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)
    BLOCKS_PER_GRID_X = (R + THREADS_PER_BLOCK_X - 1) // THREADS_PER_BLOCK_X
    BLOCKS_PER_GRID_Y = (Q + THREADS_PER_BLOCK_Y - 1) // THREADS_PER_BLOCK_Y
    BLOCKS_PER_GRID = (BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y)

    @cuda.jit
    def _kernel(
        rspec,
        qspec,
        lens,
        out,
    ):
        """
        CUDA kernel function that will be translated to GPU-executable machine code on the fly.

        Parameters:
        -----------
        rspec : cuda.devicearray
            Array containing reference spectra data.
        qspec : cuda.devicearray
            Array containing query spectra data.
        lens : cuda.devicearray
            Array containing lengths of spectra.
        out : cuda.devicearray
            Array for storing similarity scores.

        Notes:
        ------
        The kernel is designed to efficiently compute cosine similarity scores
        between reference and query spectra using CUDA parallelization.

        It performs the following steps:
        1. Compute the norms of the reference and query spectra.
        2. Find potential peak matches between the spectra based on m/z tolerance.
        3. Sort matches based on cosine product value.
        4. Accumulate cosine score, while discarding duplicate peaks.
        """
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

        # Get actual length of R
        rleni = lens[0, i]

        # We are in the grid, but current ij pair might be just a padding.
        # We check that and exit quickly if so. 
        # NOTE: We don't exit for q being zero yet! Current thread might have 
        # to do work for summing the norm of R first!
        if rleni == 0:
            return

        # We unpack mz and int from arrays
        rmz = rspec[0] 
        rint = rspec[1]
        spec1_mz = rmz[i]
        spec1_int = rint[i]
        
        #### PART 1: CALCULATE SCORE NORMS ####
        # All the threads in one block work on one reference vs. many query, so we only calculate reference norm once and share it.
        # we do this by declaring shared array with one element
        score_norm_spec1_sh = cuda.shared.array(1, types.float32)
        score_norm_spec1_sh[0] = 0
        num_active_threads_sh = cuda.shared.array(1, types.int32)
        num_active_threads_sh[0] = 0
        cuda.syncthreads()
        cuda.atomic.add(num_active_threads_sh, 0, 1) # we need to calculate how many threads are alive, since threads outside of RxQ got return'd
        cuda.syncthreads()
        num_active_threads = num_active_threads_sh[0]
        window = (rleni + num_active_threads - 1) // num_active_threads # the amount of work each thread has to do
        norm_accum = 0.0
        for ix in range(thread_j * window, min(thread_j * window + window, rleni)):
            norm_accum += (spec1_mz[ix] ** mz_power * spec1_int[ix] ** int_power) ** 2
        cuda.atomic.add(score_norm_spec1_sh, 0, norm_accum) # sum values, serially, pretty slow, but negligible to latency of the entire kernel
        cuda.syncthreads()
        score_norm_spec1 = score_norm_spec1_sh[0]
        
        # Now, with work done, threads that have padding Q can exit peacefully.
        qlenj = lens[1, j]
        if qlenj == 0:
            return
        
        qmz = qspec[0]
        qint = qspec[1]
        spec2_mz = qmz[j]
        spec2_int = qint[j]

        # Since Qs are different, each thread calculates their own norm factor for Q
        score_norm_spec2 = types.float32(0.0)
        for ix in range(qlenj):
            score_norm_spec2 += (spec2_mz[ix] ** mz_power * spec2_int[ix] ** int_power) ** 2
        score_norm = math.sqrt(score_norm_spec1) * math.sqrt(score_norm_spec2)
        spec1_mz_sh = spec1_mz # `spec1_mz_sh` is named so because at some point I experimented with R residing in shared-mem. It didn't work too well.
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
                        # Binary trick! We pack two 16bit ints in 32bit int to use less memory
                        # since we know that largest imaginable peak index can fit in 13 bits
                        matches[num_match] = (peak1_idx << 16) | peak2_idx
                        values[num_match] = (mz_r ** mz_power * int_r ** int_power) * (mz_q ** mz_power * int_q ** int_power)
                        num_match += 1
                        overflow = num_match >= MATCH_LIMIT  # This is the errorcode for overflow

        out[2, i, j] = overflow
        
        if num_match == 0:
            return

        # Debug checkpoint
        # out[i, j, 0] = score_norm
        # out[i, j, 1] = num_match
        # return

        #### PART: 3 ####
        # We use as non-recursive mergesort to order matches by the cosine product
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

        #### PART: 4 ####
        # Accumulate and deduplicate matches from largest to smallest ####
        used_r = cuda.local.array(N_MAX_PEAKS, types.boolean)
        used_q = cuda.local.array(N_MAX_PEAKS, types.boolean)
        for m in range(N_MAX_PEAKS):
            used_r[m] = False
            used_q[m] = False

        used_matches = 0
        score = 0.0
        for m in range(num_match):
            # Here we undo the binary trick
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
        # TODO: 
        # In the future, we should compile both kernels, compare perfs and use fastest kernel.
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
        rspec_cu: cuda.devicearray,
        qspec_cu: cuda.devicearray,
        lens_cu: cuda.devicearray,
        out_cu: cuda.devicearray,
        stream: cuda.stream = None,
    ) -> None:
        """
        Launches the CUDA kernel for calculating cosine similarity scores.

        Parameters:
        -----------
        rspec_cu : cuda.devicearray
            Array containing reference spectra data.
        qspec_cu : cuda.devicearray
            Array containing query spectra data.
        lens_cu : cuda.devicearray
            Array containing lengths of spectra.
        out_cu : cuda.devicearray
            Array for storing similarity scores.
        stream : cuda.stream, optional
            CUDA stream for asynchronous execution, by default None.
        """
        _kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK, stream](
            rspec_cu,
            qspec_cu,
            lens_cu,
            out_cu,
        )
    return kernel