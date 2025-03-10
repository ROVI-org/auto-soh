"""Test the code for R0 estimation"""
from pytest import fixture, raises
import numpy as np

from moirae.extractors.ecm import R0Extractor
from moirae.models.ecm.utils import unrealistic_fake_r0


@fixture()
def r0_extractor():
    return R0Extractor(10, dt_max=10.1)


@fixture()
def r0_dataset(timeseries_dataset_hppc):
    """Compute features needed for R0 extraction"""
    return timeseries_dataset_hppc


def test_data_check(r0_dataset, r0_extractor):
    r0_extractor.check_data(r0_dataset)

    # Fail if no cycle samples the full SOC
    r0_extractor.capacity *= 2
    with raises(ValueError, match='Dataset must sample 95'):
        r0_extractor.check_data(r0_dataset)


def test_spline_fit(r0_dataset, r0_extractor):
    r0_points = r0_extractor.interpolate_r0(r0_dataset.tables['raw_data'])

    expected_r0 = unrealistic_fake_r0(r0_extractor.soc_points)
    assert np.allclose(r0_points, expected_r0, atol=1e-2), f'Max diff: {np.abs(r0_points - expected_r0).max():.2e}'


def test_full(r0_dataset, r0_extractor):
    r0 = r0_extractor.extract(r0_dataset)
    assert np.isclose(r0.get_value(0.), unrealistic_fake_r0(0.), atol=1e-2)
