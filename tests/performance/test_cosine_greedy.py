import pytest
import numpy as np
from cudams.similarity import CudaCosineGreedy

@pytest.mark.performance
def test_performance(
    gnps: list,
):
    batch_size = 1024
    kernel = CudaCosineGreedy(batch_size=batch_size,
                              match_limit=1024,
                              verbose=False)
    load = np.load('tests/data/gnps_expected.npz')
    t = 0
    n = 2
    for _ in range(n):
        result = kernel.matrix(
            gnps[:batch_size],
            gnps[:batch_size],
        )
        # np.save('tests/data/gnps_expected.npz', result)
        t += kernel.kernel_time / n
    print(t)
    acc = np.isclose(load['score'], result['score']).mean()*100
    match_acc = np.isclose(load['matches'], result['matches']).mean()*100
    assert acc == 100 and match_acc == 100, f"{acc}, {match_acc}"
