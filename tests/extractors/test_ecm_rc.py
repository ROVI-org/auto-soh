"""Test the code for RC extraction"""
from pytest import fixture, raises
import numpy as np

from battdat.postprocess.integral import StateOfCharge
from moirae.extractors.ecm import RCExtractor
from moirae.models.ecm.utils import unrealistic_fake_rc


@fixture()
def rc_extractor():
    return RCExtractor(10, n_rc=2)


@fixture()
def rc_dataset(timeseries_dataset_hppc_rc):
    """Compute features needed for RC extraction"""
    StateOfCharge().enhance(timeseries_dataset_hppc_rc.tables['raw_data'])
    return timeseries_dataset_hppc_rc


def test_data_check(rc_dataset, rc_extractor):
    rc_extractor.check_data(rc_dataset)

    # Fail if no cycle samples the full SOC
    rc_extractor.capacity *= 2
    with raises(ValueError, match='Dataset rests must sample 95'):
        rc_extractor.check_data(rc_dataset)


def test_spline_fit(rc_dataset, rc_extractor):
    rc_points = rc_extractor.interpolate_rc(rc_dataset)

    expected_rc = unrealistic_fake_rc(rc_extractor.soc_points)

    for ii in range(2):

        Rdiff = f'R{ii} max diff: {np.abs(rc_points[ii][0] - expected_rc[ii][0]).max():.2e}'
        print(rc_points[ii][0])
        assert np.allclose(rc_points[ii][0], expected_rc[ii][0], atol=1e-3), Rdiff

        Cdiff = f'C{ii} max diff: {np.abs(rc_points[ii][1] - expected_rc[ii][1]).max():.2e}'
        assert np.allclose(rc_points[ii][1], expected_rc[ii][1], atol=5e2), Cdiff


def test_full(rc_dataset, rc_extractor):
    rcs = rc_extractor.extract(rc_dataset)

    for ii in range(2):
        print(rcs[ii].get_value(0.)[1], unrealistic_fake_rc(0.)[ii][1])
        assert np.isclose(rcs[ii].get_value(0.)[0], unrealistic_fake_rc(0.)[ii][0].item(), atol=1e-3)
        assert np.isclose(rcs[ii].get_value(0.)[1], unrealistic_fake_rc(0.)[ii][1].item(), atol=5e2)
