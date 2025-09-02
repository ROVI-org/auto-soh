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
                                                     gamma=h0_gt.gamma.item(),
                                                     coulombic_efficiency=ce_gt,
                                                     series_resistance=r0_gt, 
                                                     voltage_limits=voltage_lims,
                                                     max_C_rate=0.2,  # C/5 rate just to be sure
                                                     voltage_tolerance=0.0075  # 7.5 mV tolerance
                                                     )
    h0_info = extractor.extract(data=cap_check, start_soc=cap_check['SOC'].iloc[0])
    h0_ext = h0_info['value']
    soc_ext = h0_info['soc_level']
    exp_fact = np.array(h0_info['exponential_factor'])
    # Recall that the comparison will not be perfect as hysteresis takes a while to kick in, so instead, we should
    # compute the errors and weight them by the appropriate value.
    errs = (h0_ext - h0_gt.get_value(soc=soc_ext).flatten())
    mae = np.average(abs(errs), weights=exp_fact)
    rmse = np.sqrt(np.average(np.pow(errs, 2), weights=exp_fact))
    assert mae <= 0.0075, f'MAE = {mae:.1e} V > 7.5 mV!'
    assert rmse <= 0.01, f'RMSE = {rmse:.1e} V > 10 mV!'
