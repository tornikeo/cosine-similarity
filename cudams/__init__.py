import os
if os.getenv('NUMBA_ENABLE_CUDASIM') == '1':
    # Fix this issue https://github.com/numba/numba/pull/9458
    from contextlib import contextmanager
    import numba.cuda.simulator.cudadrv
    @contextmanager
    def pinned(*arylist):
        yield
    numba.cuda.simulator.cudadrv.pinned = pinned