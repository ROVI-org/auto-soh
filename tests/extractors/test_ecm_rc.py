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


def test_synthetic_realistic_hppc(realistic_rpt_data) -> None:
    """
    Function that tests the RC extractor on a more realistic HPPC profile from synthetic data representative of an
    LFP cell
    """
    # Get data
    raw_data = realistic_rpt_data.tables['raw_data']

    # The capacity of this cell is 30 Amp-hour
    capacity = 30

    # Now, let's use the RC extractor on the HPPC data, which corresponds to the second cycle (the first one was the
    # low C-rate capacity check)
    hppc_data = raw_data[raw_data['cycle_number'] == 2]
    # In this synthetic data, only one RC component is present. The cycle starts at 100% SOC, as it is a discharge HPPC
    rc_ext = RCExtractor(capacity=capacity,
                         starting_soc=1.0,
                         soc_points=1,
                         n_rc=1)
    rc_comp = rc_ext.extract_from_raw(data=hppc_data)

    # The resistance should be 1.5e-03 Ohms, and the capacitance, 2.0e+05 Farads
    expected_r = 1.5e-03
    expected_c = 2.0e+05
    assert np.allclose(rc_comp[0].r.base_values, expected_r, rtol=0.05), \
        f'Wrong resistance base values: {rc_comp[0].r.base_values}'
    assert np.allclose(rc_comp[0].c.base_values, expected_c, rtol=0.05), \
        f'Wrong capacitance base values: {rc_comp[0].c.base_values}'
