"""Test the code for OCV estimation"""
from battdat.postprocess.integral import StateOfCharge
from pytest import fixture
import numpy as np

from moirae.extractor.ecm import OCVExtractor
from moirae.models.ecm.utils import realistic_fake_ocv


@fixture()
def ocv_dataset(timeseries_dataset):
    """Compute features needed for OCV extraction"""
    StateOfCharge().enhance(timeseries_dataset.tables['raw_data'])
    return timeseries_dataset


def test_cycle_detection(ocv_dataset):
    cycle = OCVExtractor().find_best_cycle(ocv_dataset)
    assert cycle['cycle_number'].unique().size == 1


def test_spline_fit(ocv_dataset):
    extractor = OCVExtractor()
    cycle = ocv_dataset.tables['raw_data']
    ocv_points = extractor.interpolate_ocv(ocv_dataset, cycle)

    expected_ocv = realistic_fake_ocv(extractor.soc_points)
    assert np.allclose(ocv_points, expected_ocv, atol=1e-1), f'Max diff: {np.abs(ocv_points - expected_ocv).max():.2e}'


def test_full(ocv_dataset):
    ocv = OCVExtractor().extract(ocv_dataset)
    assert np.isclose(ocv(0.), realistic_fake_ocv(0.), atol=1e-2)
