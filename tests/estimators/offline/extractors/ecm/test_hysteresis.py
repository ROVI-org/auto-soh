"""
Tests hysteresis extraction
"""
import numpy as np

from moirae.estimators.offline.extractors.ecm import HysteresisExtractor


def test_realistic_synthetic(realistic_LFP_aSOH, realistic_rpt_data):
    # Get the capacity check
    raw_rpt = realistic_rpt_data.tables.get('raw_data')
    cap_check = raw_rpt[raw_rpt['protocol'] == 'Capacity Check']

    # For simplicity, get the ground truth OCV and other relevant values
    qt_gt = realistic_LFP_aSOH.q_t
    r0_gt = realistic_LFP_aSOH.r0
    ce_gt = realistic_LFP_aSOH.ce
    ocv_gt = realistic_LFP_aSOH.ocv
    h0_gt = realistic_LFP_aSOH.h0
    voltage_lims = tuple(ocv_gt.get_value(soc=[0., 1.]).flatten().tolist())

    # Prepare extractor
    extractor = HysteresisExtractor.init_from_basics(capacity=qt_gt,
                                                     ocv=ocv_gt,
                                                     coulombic_efficiency=ce_gt,
                                                     series_resistance=r0_gt, 
                                                     voltage_limits=voltage_lims,
                                                     max_C_rate=0.2,  # C/5 rate just to be sure
                                                     voltage_tolerance=0.0075  # 7.5 mV tolerance
                                                     )
    h0_info = extractor.extract(data=cap_check, start_soc=cap_check['SOC'].iloc[0])
    h0_ext = h0_info['value']
    soc_ext = h0_info['soc_level']
    step_time = h0_info['step_time']
    # Recall that the comparison will not be perfect as hysteresis takes a while to kick in, so instead, we should
    # compute the errors and weight them by the appropriate step time value.
    gamma = h0_gt.gamma.flatten()
    kappa = gamma * realistic_LFP_aSOH.ce.flatten() / qt_gt.value.flatten()  # this is the rate of decay
    # Compute amount of time it takes to reach 98% of the max hyst value, knowing exp(-4) ~ 0.02
    time_to_98_percent = 4. / kappa  
    weights = step_time / time_to_98_percent
    weights = np.where(weights > 1., 1., weights)
    errs = (h0_ext - h0_gt.get_value(soc=soc_ext).flatten()) * weights
    assert np.allclose(errs, 0.0, atol=0.015), f'Max weighted hyst error = {max(abs(errs))}'
