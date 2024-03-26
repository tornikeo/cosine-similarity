from pathlib import Path
import pytest
import numpy as np
from cudams.similarity import CudaCosineGreedy


@pytest.mark.parametrize(
    'batch_size, match_limit, array_type, sparse_threshold', 
    [
        [1024, 1024, 'numpy', 0,],
        [1024, 2048, 'numpy', 0,],
        [2048, 1024, 'numpy', 0,],
        [1024, 1024, 'sparse', .75],
    ]
)
@pytest.mark.performance
def test_performance(
    gnps: list,
    batch_size: int,
    match_limit: int,
    array_type: str,
    sparse_threshold: float,
):
    kernel = CudaCosineGreedy(batch_size=batch_size,
                              match_limit=match_limit, 
                              sparse_threshold=sparse_threshold,
                              verbose=False)
    n = 3
    # Warm-up
    kernel.matrix(gnps[:8], gnps[:8])
    times = []
    for _ in range(n):
        kernel.matrix(
            gnps[:batch_size],
            gnps[:batch_size],
            array_type=array_type,
        )
        times.append(kernel.kernel_time)
    times = np.array(times)
    print(f"\n=> PERF:  {times.mean():.4f}s +-{times.std():.4f}std @ {batch_size}, {match_limit}, {array_type}, {sparse_threshold}, \n")