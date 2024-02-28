from matchms.similarity import BaseSimilarity
from matchms import Spectrum
from typing import List
import numpy as np
import pytest
from cudams.similarity import CudaCosineGreedy, CudaFingerprintSimilarity
from matchms.similarity import CosineGreedy, FingerprintSimilarity
from ..builder_Spectrum import SpectrumBuilder
from matchms import calculate_scores, Scores

def equality_function_cosine_greedy(scores_cu: Scores, scores: Scores):
    score = scores[f'CosineGreedy_score']
    score_cu = scores_cu[f'CudaCosineGreedy_score']
    assert np.allclose(score, score_cu, equal_nan=True)
    
    matches = scores['CosineGreedy_matches']
    matches_cu = scores_cu['CudaCosineGreedy_matches']
    assert np.allclose(matches, matches_cu, equal_nan=True)
    

def equality_function_fingerprint(scores: Scores, scores_cu: Scores, ):
    assert np.allclose(scores, scores_cu, equal_nan=True)

@pytest.mark.parametrize(
        "similarity_function, args, cuda_similarity_function, cu_args, equality_function", 
    [
        (FingerprintSimilarity, ('jaccard',),   CudaFingerprintSimilarity, ('jaccard',),    equality_function_fingerprint),
        (FingerprintSimilarity, ('cosine',),    CudaFingerprintSimilarity, ('cosine',),     equality_function_fingerprint),
        (FingerprintSimilarity, ('dice',),      CudaFingerprintSimilarity, ('dice',),       equality_function_fingerprint),
        (CosineGreedy,          (),             CudaCosineGreedy,          (),              equality_function_cosine_greedy),
    ]
)
def test_compatibility(
        gnps_library_256: List[Spectrum],
        
        similarity_function: BaseSimilarity,
        args: tuple,
        
        cuda_similarity_function: BaseSimilarity,
        cu_args: tuple,
        
        equality_function: callable,
    ):
    references, queries = gnps_library_256, gnps_library_256
    similarity_function = similarity_function(*args)
    scores = calculate_scores(
        references=references,
        queries=queries,
        similarity_function=similarity_function,
        is_symmetric=True
    ).to_array()
    cuda_similarity_function = cuda_similarity_function(*cu_args)
    scores_cu = calculate_scores(
        references=references,
        queries=queries,
        similarity_function=cuda_similarity_function
    ).to_array()
    
    equality_function(scores, scores_cu)
    
    # try:    
    #     overflow_cu = scores_cu[f'{similarity_function.__class__.__name__}_overflow']
    #     assert np.allclose(matches, matches_cu)
    # except KeyError:
    #     pass
    