import os
from contextlib import contextmanager

# Patch cuda.pinned before https://github.com/numba/numba/pull/9458 gets merged
if os.getenv('NUMBA_ENABLE_CUDASIM') == '1':
    from numba import cuda
    @contextmanager
    def fake_cuda_pinned(*arylist):
        yield
    cuda.pinned = fake_cuda_pinned
