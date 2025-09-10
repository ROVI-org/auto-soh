"""Test the code for R0 estimation"""
from pytest import fixture, raises
import numpy as np

from moirae.estimators.offline.DataCheckers.base import DataCheckError
from moirae.estimators.offline.extractors.ecm import R0Extractor
from moirae.models.ecm.utils import unrealistic_fake_r0


@fixture()
def r0_extractor():
    return R0Extractor.init_from_basics(capacity=10, dt_max=10.1, min_delta_soc=0.95)


@fixture()
def r0_dataset(timeseries_dataset_hppc):
    """Compute features needed for R0 extraction"""
    return timeseries_dataset_hppc


def test_data_check(r0_dataset, r0_extractor):
    r0_extractor.data_checker.check(r0_dataset)

    # Fail if no cycle samples the full SOC
    r0_extractor.capacity = 2 * r0_extractor.capacity
    with raises(DataCheckError, match='Dataset must sample at least 95'):
        r0_extractor.data_checker.check(r0_dataset)


def test_spline_fit(r0_dataset, r0_extractor):
    r0_info = r0_extractor.compute_parameters(r0_dataset.tables['raw_data'])
    r0_vals = r0_info['value']

    expected_r0 = unrealistic_fake_r0(r0_info['soc_level'])
    assert np.allclose(r0_vals, expected_r0, atol=1e-2), f'Max diff: {np.abs(r0_vals - expected_r0).max():.2e}'


def test_full_simple(r0_dataset, r0_extractor):
    r0_info = r0_extractor.extract(r0_dataset)
    r0_vals = r0_info['value']
    expected_r0 = unrealistic_fake_r0(r0_info['soc_level'])
    assert np.allclose(r0_vals, expected_r0, atol=1e-2), f'Max diff: {np.abs(r0_vals - expected_r0).max():.2e}'


def test_full_realistic(realistic_rpt_data, realistic_LFP_aSOH):
    # Get the relevant part of the data
    raw_rpt = realistic_rpt_data.tables.get('raw_data')
    hppc = raw_rpt[raw_rpt['protocol'] == 'Full HPPC']

    # Prepare what is needed
    capacity = realistic_LFP_aSOH.q_t
    coul_eff = realistic_LFP_aSOH.ce
    r0_extractor = R0Extractor.init_from_basics(capacity=capacity,
                                                coulombic_efficiency=coul_eff,
                                                min_delta_soc=0.99,
                                                min_pulses=10,
                                                ensure_bidirectional=True,
                                                min_number_of_rests=10)

    # Recall we start at close to 100% SOC
    start_soc = hppc.iloc[0]['SOC']

    # Extract
    r0_info = r0_extractor.extract(data=hppc, start_soc=start_soc)
    r0_vals = r0_info['value']
    expected_r0 = realistic_LFP_aSOH.r0.get_value(soc=r0_info['soc_level'])
    assert np.allclose(r0_vals, expected_r0, atol=1e-2), f'Max diff: {np.abs(r0_vals - expected_r0).max():.2e}'
