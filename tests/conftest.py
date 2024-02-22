import pytest, warnings

@pytest.fixture(autouse=True, scope='module')
def use_cudasim_if_cuda_unavailable():
    import os, numba
    
    os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
    
    has_cuda = numba.cuda.is_available()
    
    if not has_cuda:
        warnings.warn("CUDA was unavailable - using numba cuda simulator for testing")
        os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

    yield
    
    if not has_cuda:    
        del os.environ['NUMBA_ENABLE_CUDASIM']
        del os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS']