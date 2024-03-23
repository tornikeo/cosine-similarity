import pytest
import numpy as np
from cudams.similarity import CudaCosineGreedy

@pytest.mark.performance
def test_performance(
    gnps: list,
):
    batch_size = 1024
    kernel = CudaCosineGreedy(batch_size=batch_size,
                              max_spectra_length=1024,
                              match_limit=1024,
                              verbose=False)
    load = np.load('tests/data/gnps_expected.npz')
    t = 0
    n = 3
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
    assert acc == 100 and match_acc == 100, (f"{acc}, {match_acc}, \n" 
                                            f" {result['score'].mean()}\t{result['score'].max()}\t{result['score'].min()} \n"
                                            f" {result['matches'].mean()}\t{result['matches'].max()}\t{result['matches'].min()} \n"
                                            f" {result['overflow'].mean()}\t{result['overflow'].max()}\t{result['overflow'].min()} \n"
                                            '  EXPECTED STATS \n'
                                            f" {load['score'].mean()}\t{load['score'].max()}\t{load['score'].min()} \n"
                                            f" {load['matches'].mean()}\t{load['matches'].max()}\t{load['matches'].min()} \n"
                                        )
