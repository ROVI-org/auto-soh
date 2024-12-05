"""Test the code for OCV estimation"""
from battdat.postprocess.integral import StateOfCharge
from pytest import fixture, raises
import numpy as np

from moirae.extractors.ecm import OCVExtractor
from moirae.models.ecm.utils import realistic_fake_ocv


@fixture()
def ocv_extractor():
    return OCVExtractor(10)


@fixture()
def ocv_dataset(timeseries_dataset):
    """Compute features needed for OCV extraction"""
    StateOfCharge().enhance(timeseries_dataset.tables['raw_data'])
    return timeseries_dataset


def test_data_check(ocv_dataset, ocv_extractor):
    ocv_extractor.check_data(ocv_dataset)

    # Fail if no cycle samples the full SOC
    ocv_extractor.capacity *= 2
    with raises(ValueError, match='Dataset must sample 95'):
        ocv_extractor.check_data(ocv_dataset)


def test_spline_fit(ocv_dataset, ocv_extractor):
    ocv_points = ocv_extractor.interpolate_ocv(ocv_dataset.tables['raw_data'])

    expected_ocv = realistic_fake_ocv(ocv_extractor.soc_points)
    assert np.allclose(ocv_points, expected_ocv, atol=1e-1), f'Max diff: {np.abs(ocv_points - expected_ocv).max():.2e}'


def test_full(ocv_dataset, ocv_extractor):
    ocv = ocv_extractor.extract(ocv_dataset)
    assert np.isclose(ocv(0.), realistic_fake_ocv(0.), atol=1e-2)
