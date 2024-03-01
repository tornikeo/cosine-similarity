from cudams.utils import Timer
import numpy as np
import pytest
from cudams.similarity import CudaCosineGreedy
from ..builder_Spectrum import SpectrumBuilder
import os

@pytest.mark.skipif(
    os.getenv("PERFTEST") != "1",
    reason="Github workflows isn't the best tool for performance measurement.",
)
def test_performance(
    gnps_library_8k: list,
):
    kernel = CudaCosineGreedy(batch_size=2048,
                              match_limit=128, 
                              verbose=True)
    with Timer() as timer:
        kernel.matrix(
            gnps_library_8k,
            gnps_library_8k,
            array_type='sparse', 
            sparse_threshold=.75,
        )
    assert timer.duration < 60