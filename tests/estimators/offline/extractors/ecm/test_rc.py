"""Test the code for RC extraction"""
from pytest import fixture, raises
import numpy as np

from battdat.postprocess.integral import StateOfCharge

from moirae.estimators.offline.DataCheckers.base import DataCheckError
from moirae.estimators.offline.extractors.ecm import MaxCapacityCoulEffExtractor, RCExtractor
from moirae.models.ecm.utils import unrealistic_fake_rc


@fixture()
def rc_extractor() -> RCExtractor:
    return RCExtractor.init_from_basics(capacity=10, min_delta_soc=0.95)


@fixture()
def rc_dataset(timeseries_dataset_hppc_rc):
    """Compute features needed for RC extraction"""
    # StateOfCharge().enhance(timeseries_dataset_hppc_rc.tables['raw_data'])
    return timeseries_dataset_hppc_rc


def test_data_check(rc_dataset, rc_extractor):
    rc_extractor.data_checker.check(rc_dataset)

    # Fail if no cycle samples the full SOC
    rc_extractor.capacity = 2 * rc_extractor.capacity
    with raises(DataCheckError, match='Dataset must sample at least 95'):
        rc_extractor.data_checker.check(rc_dataset)


def test_spline_fit(rc_dataset, rc_extractor):
    rc_extracted = rc_extractor.extract(rc_dataset, n_rc=2)
    assert len(rc_extracted) == 2, f'Extracted different number of RC elements: {rc_extracted}'

    expected_rc = unrealistic_fake_rc(rc_extracted[0][0]['soc_level'])

    for ii in range(2):
        r_vals = rc_extracted[ii][0]['value']
        c_vals = rc_extracted[ii][1]['value']
        Rdiff = f'R{ii} max diff: {np.abs(r_vals - expected_rc[ii][0]).max():.2e}'
        assert np.allclose(r_vals, expected_rc[ii][0], atol=1e-3), Rdiff

        Cdiff = f'C{ii} max diff: {np.abs(c_vals - expected_rc[ii][1]).max():.2e}'
        assert np.allclose(c_vals, expected_rc[ii][1], atol=5e2), Cdiff


def test_synthetic_realistic_hppc(realistic_rpt_data, realistic_LFP_aSOH) -> None:
    """
    Function that tests the RC extractor on a more realistic HPPC profile from synthetic data representative of an
    LFP cell
    """
    # Get data
    raw_data = realistic_rpt_data.tables['raw_data']

    # The capacity of this cell is 30 Amp-hour
    ground_truth_capacity = realistic_LFP_aSOH.q_t.amp_hour.item()
    capacity_check = raw_data[raw_data['protocol'] == 'Capacity Check']
    extracted_capacity_info, _ = MaxCapacityCoulEffExtractor().extract(data=capacity_check)
    extracted_cap = extracted_capacity_info['value']
    assert np.allclose(extracted_cap, ground_truth_capacity, rtol=0.01), \
        f'Wrong capacity! Expected {ground_truth_capacity}, extracted {extracted_cap}'

    # Now, let's use the RC extractor on the HPPC data, which corresponds to the second cycle (the first one was the
    # low C-rate capacity check)
    hppc_data = raw_data[raw_data['protocol'] == 'Full HPPC']
    # In this synthetic data, only one RC component is present. The cycle starts at 100% SOC, as it is a discharge HPPC
    extractor = RCExtractor.init_from_basics(capacity=extracted_cap)
    rc_info = extractor.extract(data=hppc_data, start_soc=1.0, n_rc=1)

    # The resistance should be 1.5e-03 Ohms, and the capacitance, 2.0e+05 Farads
    expected_r = 1.5e-03
    expected_c = 2.0e+05
    obtained_r = rc_info[0][0]['value']
    obtained_c = rc_info[0][1]['value']
    assert np.allclose(obtained_r, expected_r, rtol=0.05), \
        f'Wrong resistance values: {obtained_r}'
    assert np.allclose(obtained_c, expected_c, rtol=0.05), \
        f'Wrong capacitance values: {obtained_c}'
