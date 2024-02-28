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
    
@pytest.fixture(autouse=True)
def patch_harmonize_values(monkeypatch):
    """
    Necessary until https://github.com/matchms/matchms/pull/605 gets merged
    """
    from matchms import Metadata
    from matchms.filtering.metadata_processing.add_precursor_mz import \
        _add_precursor_mz_metadata
    from matchms.filtering.metadata_processing.add_retention import (
        _add_retention, _retention_index_keys, _retention_time_keys)
    from matchms.filtering.metadata_processing.interpret_pepmass import \
        _interpret_pepmass_metadata
    from matchms.filtering.metadata_processing.make_charge_int import \
        _convert_charge_to_int

    def harmonize_values(self):
        """Runs default harmonization of metadata.

        This includes harmonizing entries for ionmode, retention time and index,
        charge, as well as the removal of invalid entried ("", "NA", "N/A", "NaN").
        """
        metadata_filtered = _interpret_pepmass_metadata(self.data)
        metadata_filtered = _add_precursor_mz_metadata(metadata_filtered)

        if metadata_filtered.get("ionmode"):
            metadata_filtered["ionmode"] = self.get("ionmode").lower()

        if metadata_filtered.get("retention_time"):
            metadata_filtered = _add_retention(metadata_filtered, "retention_time", _retention_time_keys)

        if metadata_filtered.get("retention_index"):
            metadata_filtered = _add_retention(metadata_filtered, "retention_index", _retention_index_keys)

        if metadata_filtered.get("parent"):
            metadata_filtered["parent"] = float(metadata_filtered.get("parent"))

        charge = metadata_filtered.get("charge")
        charge_int = _convert_charge_to_int(charge)
        if not isinstance(charge, int) and charge_int is not None:
            metadata_filtered["charge"] = charge_int

        invalid_entries = ["", "NA", "N/A", "NaN"]

        
        metadata_filtered_ = {}
        # Necessary to check not isinstance(..., str), since some values are arrays, and `not in`
        # operator results in iterable, that has an ambiguous truth value
        for k,v in metadata_filtered.items():
            if not isinstance(v, str) or v not in invalid_entries:
                metadata_filtered_[k] = v
        self.data = metadata_filtered_

    monkeypatch.setattr(Metadata, 'harmonize_values', harmonize_values)
    yield
    

@pytest.fixture(scope='session')
def gnps_library_512():
    from cudams.utils import download
    from matchms.filtering import default_filters, normalize_intensities, reduce_to_number_of_peaks
    from matchms.importing import load_from_mgf
    from joblib import Parallel, delayed
    
    fpath = download('GNPS-LIBRARY.mgf')

    def parse_spectrum(spectrum):
        spectrum = default_filters(spectrum)
        spectrum = reduce_to_number_of_peaks(spectrum, n_max=1024)
        spectrum = normalize_intensities(spectrum)
        return spectrum
    
    spectra = tuple(sp for _, sp in zip(range(1024), load_from_mgf(fpath)))
    spectra = Parallel(-2)(delayed(parse_spectrum)(s) for s in spectra)
    spectra = [spe for spe in spectra if spe is not None]
    spectra = spectra[:512]
    yield spectra
    

@pytest.fixture(scope='session')
def gnps_library_256():
    from cudams.utils import download
    from matchms.filtering import default_filters, normalize_intensities, reduce_to_number_of_peaks, add_fingerprint
    from matchms.importing import load_from_mgf
    from joblib import Parallel, delayed
    
    fpath = download('GNPS-LIBRARY.mgf')

    def parse_spectrum(spectrum):
        spectrum = default_filters(spectrum)
        spectrum = reduce_to_number_of_peaks(spectrum, n_max=1024)
        spectrum = normalize_intensities(spectrum)
        return spectrum
    
    spectra = tuple(sp for _, sp in zip(range(1024), load_from_mgf(fpath)))
    spectra = Parallel(-2)(delayed(parse_spectrum)(s) for s in spectra)
    spectra = [spe for spe in spectra if spe is not None]
    spectra = spectra[:512]
    spectra = [add_fingerprint(x, nbits=256) for x in spectra] 
    yield spectra
    