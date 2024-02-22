import pytest, warnings

@pytest.fixture(autouse=True, scope='session')
def warn_on_no_cuda():
    import os, numba
    if not numba.cuda.is_available():
        warnings.warn("CUDA was unavailable - consider using `NUMBA_ENABLE_CUDASIM=1 pytest <same args, if any>` to simulate having GPU and cudatoolkit for testing purposes")
    yield
        
@pytest.fixture(autouse=True, scope='session')
def ignore_warnings():
    import os
    os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'
    yield