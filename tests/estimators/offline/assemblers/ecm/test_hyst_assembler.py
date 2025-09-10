"""
Tests hysteresis assembler
"""
from pytest import raises
import numpy as np

from moirae.estimators.offline.assemblers.utils import SOCRegressor
from moirae.estimators.offline.extractors.ecm import HysteresisExtractor
from moirae.estimators.offline.assemblers.ecm import HysteresisAssembler


def test_wrong_units():
    hyst_ass = HysteresisAssembler()
    with raises(ValueError, match='OCV provided in Volts, rather than Volt!'):
        _ = hyst_ass.assemble(extracted_parameter={'units': 'Volts',
                                                   'value': [0., 1.],
                                                   'soc_level': [0., 1.],
                                                   'adjusted_curr': [1., 1.],
                                                   'step_time': [0., 1.]})


def test_proper_assembly(realistic_rpt_data, realistic_LFP_aSOH):
    # Get the capacity check
    raw_rpt = realistic_rpt_data.tables.get('raw_data')
    cap_check = raw_rpt[raw_rpt['protocol'] == 'Capacity Check']

    # For simplicity, get the ground truth OCV and other relevant values
    qt_gt = realistic_LFP_aSOH.q_t
    r0_gt = realistic_LFP_aSOH.r0
    ce_gt = realistic_LFP_aSOH.ce
    ocv_gt = realistic_LFP_aSOH.ocv
    rc_gt = realistic_LFP_aSOH.rc_elements
    h0_gt = realistic_LFP_aSOH.h0
    voltage_lims = tuple(ocv_gt.get_value(soc=[0., 1.]).flatten().tolist())

    # Prepare extractor
    extractor = HysteresisExtractor.init_from_basics(capacity=qt_gt,
                                                     coulombic_efficiency=ce_gt,
                                                     ocv=ocv_gt,
                                                     series_resistance=r0_gt,
                                                     rc_elements=rc_gt,
                                                     voltage_limits=voltage_lims,
                                                     max_C_rate=0.2,  # C/5 rate just to be sure
                                                     voltage_tolerance=0.0075  # 7.5 mV tolerance
                                                     )
    hyst_info = extractor.extract(data=cap_check, start_soc=cap_check['SOC'].iloc[0])

    # Assemble new hysteresis
    gt_soc_pts = h0_gt.soc_pinpoints.flatten()
    # Prepare a SOC regressor that is not based on smoothing, since the ground truth is sort of piecewise constant
    assembler = HysteresisAssembler(regressor=SOCRegressor(style='interpolate', parameters={'k': 0}),
                                    soc_points=gt_soc_pts)
    h0_ass = assembler.assemble(extracted_parameter=hyst_info)
    # Test the values
    soc_test = np.linspace(0., 1., 100)
    h0_ass_vals = h0_ass.get_value(soc_test).flatten()
    h0_gt_vals = h0_gt.get_value(soc_test).flatten()
    errs = h0_ass_vals - h0_gt_vals
    assert np.allclose(errs, 0, atol=0.01), f'Max error {max(abs(errs)):.1e} V > 10 mV!'
    assert np.mean(abs(errs)) < 2.5e-03, f'Hyst MAE {np.mean(abs(errs)):.1e} V > 2.5 mV'
    rmse = np.sqrt(np.mean(np.pow(errs, 2)))
    assert rmse < 3.5e-03, f'Hyst RMSE {rmse:.1e} V > 3.5 mV'
