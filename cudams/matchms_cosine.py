from typing import Tuple

from matchms.similarity import CosineGreedy as OriginalCosineGreedy
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.similarity.spectrum_similarity_functions import (
    collect_peak_pairs, score_best_matches)
from matchms.typing import SpectrumType


class CosineGreedy(OriginalCosineGreedy):
    """Stable implementation of original cosine greedy"""
    def __init__(self, tolerance: float = 0.1, mz_power: float = 0, intensity_power: float = 1):
        super().__init__(tolerance, mz_power, intensity_power)
        
    def pair(self, reference: SpectrumType, query: SpectrumType) -> Tuple[float, int]:
        """Calculate cosine score between two spectra.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns
        -------
        Score
            Tuple with cosine score and number of matched peaks.
        """
        def get_matching_pairs():
            """Get pairs of peaks that match within the given tolerance."""
            matching_pairs = collect_peak_pairs(spec1, spec2, self.tolerance,
                                                shift=0.0, mz_power=self.mz_power,
                                                intensity_power=self.intensity_power)
            if matching_pairs is None:
                return None
            matching_pairs = matching_pairs[np.argsort(matching_pairs[:, 2],kind='mergesort')[::-1], :]
            return matching_pairs

        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy
        matching_pairs = get_matching_pairs()
        if matching_pairs is None:
            return np.asarray((float(0), 0), dtype=self.score_datatype)
        score = score_best_matches(matching_pairs, spec1, spec2,
                                   self.mz_power, self.intensity_power)
        return np.asarray(score, dtype=self.score_datatype)
