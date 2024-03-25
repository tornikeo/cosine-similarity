from pathlib import Path
import pytest
import numpy as np
from cudams.similarity import CudaCosineGreedy


@pytest.mark.parametrize(
    'batch_size, match_limit, array_type, sparse_threshold, expected_pickle', 
    [
        [1024, 1024, 'numpy', 0, 'tests/data/gnps_expected.npz'],
        [1024, 2048, 'numpy', 0, 'tests/data/gnps_expected.npz'],
        [2048, 1024, 'numpy', 0, None],
        [1024, 1024, 'sparse', .75, None],
    ]
)
@pytest.mark.performance
def test_performance(
    gnps: list,
    batch_size: int,
    match_limit: int,
    array_type: str,
    sparse_threshold: float,
    expected_pickle: str,
):
    kernel = CudaCosineGreedy(batch_size=batch_size, match_limit=match_limit, verbose=False)
    t = 0
    n = 2
    for _ in range(n):
        result = kernel.matrix(
            gnps[:batch_size],
            gnps[:batch_size],
            array_type=array_type,
            score_threshold=sparse_threshold
        )
        # np.save('tests/data/gnps_expected.npz', result)
        t += kernel.kernel_time / n
    print(f"\n=> PERF:  {t:.3f} @ {batch_size}, {match_limit}, {array_type}, {sparse_threshold}, \n")

    if expected_pickle is not None:
        loader = np.load
        load = loader(expected_pickle)

        acc = np.isclose(load['score'], result['score']).mean()
        match_acc = np.isclose(load['matches'], result['matches']).mean()

        correct = acc == 1 and match_acc == 1
        if not correct:
            print(
                f"=== Accuracy {acc}, {match_acc} ===\n" 
                f'=== stats === \n'
                f" {result['score'].mean()}\t{result['score'].max()}\t{result['score'].min()} \n"
                f" {result['matches'].mean()}\t{result['matches'].max()}\t{result['matches'].min()} \n"
                f" {result['overflow'].mean()}\t{result['overflow'].max()}\t{result['overflow'].min()} \n"
                '=== Expected stats === \n'
                f" {load['score'].mean()}\t{load['score'].max()}\t{load['score'].min()} \n"
                f" {load['matches'].mean()}\t{load['matches'].max()}\t{load['matches'].min()} \n"
            )
            assert False
