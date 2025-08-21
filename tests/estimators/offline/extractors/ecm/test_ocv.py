"""Test the code for OCV estimation"""
from pytest import fixture, raises
import numpy as np

from battdat.data import BatteryDataset

from moirae.estimators.offline.extractors.ecm import OCVExtractor


@fixture()
def ocv_extractor():
    return OCVExtractor.init_from_basics(capacity=10,
                                         max_C_rate=2,
                                         voltage_limits=(1, 5))


@fixture()
def ocv_dataset(timeseries_dataset):
    raw_data = timeseries_dataset.tables.get('raw_data')
    cycle0 = raw_data[raw_data['cycle_number'] == 0]
    return BatteryDataset.make_cell_dataset(raw_data=cycle0)


def test_data_check(ocv_dataset, ocv_extractor):
    with raises(ValueError, match='Cycle does not reach'):
        ocv_extractor.data_checker.check(ocv_dataset)


def test_realistic_synthetic(realistic_LFP_aSOH, realistic_rpt_data):
    # Get the capacity check
    raw_rpt = realistic_rpt_data.tables.get('raw_data')
    cap_check = raw_rpt[raw_rpt['protocol'] == 'Capacity Check']

    # For simplicity, get the ground truth OCV and other relevant values
    qt_gt = realistic_LFP_aSOH.q_t
    r0_gt = realistic_LFP_aSOH.r0
    ce_gt = realistic_LFP_aSOH.ce
    ocv_gt = realistic_LFP_aSOH.ocv
    voltage_lims = tuple(ocv_gt.get_value(soc=[0., 1.]).flatten().tolist())

    # Prepare extractor
    extractor = OCVExtractor.init_from_basics(capacity=qt_gt,
                                              coulombic_efficiency=ce_gt,
                                              series_resistance=r0_gt,
                                              voltage_limits=voltage_lims,
                                              max_C_rate=0.2,  # C/5 rate just to be sure
                                              voltage_tolerance=0.0075  # 7.5 mV tolerance
                                              )
    ocv_info = extractor.extract(data=cap_check, start_soc=cap_check['SOC'].iloc[0])
    ocv_ext = ocv_info['value']
    soc_ext = ocv_info['soc_level']
    # Recall that the comparison will not be perfect due to hysteresis and RC terms that were not accounted for, as well
    # as numerical errors in the computation of SOC. A 5% error seems reasonable
    errs = ocv_ext - ocv_gt(soc=soc_ext).flatten()
    assert np.allclose(ocv_gt.get_value(soc=soc_ext).flatten(), ocv_ext, rtol=0.05), \
        f'Max OCV error = {max(abs(errs))}'
