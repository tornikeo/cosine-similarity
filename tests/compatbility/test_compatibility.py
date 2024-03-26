from typing import List
import numpy as np
import pytest
from matchms import Scores, Spectrum, calculate_scores
from matchms.similarity import (BaseSimilarity, CosineGreedy,
                                FingerprintSimilarity)
from cudams.similarity import CudaCosineGreedy, CudaFingerprintSimilarity
import pytest
import matchms
import numpy as np
from cudams.similarity import CudaCosineGreedy
from ..utils import get_expected_cosine_greedy_score


@pytest.mark.parametrize(
    'batch_size', 
    [
        512
    ]
)
def test_cosinegreedy_compat(
    gnps: list,
    batch_size: int,
):
    match_limit=1024
    r = gnps[:batch_size]
    q = gnps[:batch_size]
    
    # expected_score = get_expected_cosine_greedy_score(
    #     r, q
    # )
    # kernel = CudaCosineGreedy(batch_size=batch_size, 
    #                           match_limit=match_limit, 
    #                           verbose=False)
    # result = kernel.matrix(
    #     r, q,
    # )


    equals = np.isclose(expected_score['score'], result['score'], atol=.001)
    match_equals = np.isclose(expected_score['matches'], result['matches'])
    equals_except_overflows = equals | result['overflow']
    match_equals_except_overflows = match_equals | result['overflow']

    accuracy_rate = equals_except_overflows.mean()
    inaccuracy_num = (1-equals_except_overflows).sum()
    match_accuracy_rate = match_equals_except_overflows.mean()
    match_inaccuracy_num = (1-match_equals_except_overflows).sum()
    overflow_rate = result['overflow'].mean()
    overflow_num = result['overflow'].sum()
    
    errors = []
    warns = []
    if accuracy_rate < 1:
        errors.append(f'accuracy={accuracy_rate:.7f} # {inaccuracy_num}')
    if match_accuracy_rate < 1:
        errors.append(f'match_acc={match_accuracy_rate:.7f} # {match_inaccuracy_num}')
    if overflow_rate > 0:
        warns.append(f'overflow={overflow_rate:.7f} # {overflow_num}')
    assert not errors, f"ERR: {errors}, \n WARN: {warns}"
    # if not correct
    #     print(
    #         f"\n === Accuracy {equals}, {match_equals} ===\n" 
    #         f'=== stats === \n'
    #         f" {result['score'].mean()}\t{result['score'].max()}\t{result['score'].min()} \n"
    #         f" {result['matches'].mean()}\t{result['matches'].max()}\t{result['matches'].min()} \n"
    #         f" {result['overflow'].mean()}\t{result['overflow'].max()}\t{result['overflow'].min()} \n"
    #         '=== Expected stats === \n'
    #         f" {expected_score['score'].mean()}\t{expected_score['score'].max()}\t{expected_score['score'].min()} \n"
    #         f" {expected_score['matches'].mean()}\t{expected_score['matches'].max()}\t{expected_score['matches'].min()} \n"
    #     )
    # assert correct