"""Get the maximum capacity of a cell"""
import numpy as np

from moirae.estimators.offline.extractors.ecm import MaxCapacityCoulEffExtractor


def test_max_qt(realistic_rpt_data, realistic_LFP_aSOH):
    # Get the capacity check part of the data
    raw_rpt = realistic_rpt_data.tables.get('raw_data')
    cap_check = raw_rpt[raw_rpt['protocol'] == 'Capacity Check']

    # Get ground truth values
    qt_gt = realistic_LFP_aSOH.q_t.amp_hour.item()
    ce_gt = realistic_LFP_aSOH.ce.item()

    # Initialize extractor
    extractor = MaxCapacityCoulEffExtractor()

    # Get extracted values
    max_cap_info, ce_info = extractor.extract(data=cap_check)
    max_cap = max_cap_info['value']
    ce_ext = ce_info['value']

    # Compare, using a 1% tolerance on capacity due to hysteresis
    assert np.allclose(max_cap, qt_gt, rtol=1.0e-02), \
        f'Extracted capacity {max_cap:.1f} != {qt_gt:.1f} ground truth Amp-hours!'
    assert np.allclose(ce_gt, ce_ext), f'Extracted CE {ce_ext:.3f} != {ce_gt:.3} ground truth!'

    # Make sure other things are expected
    cap_unit = max_cap_info['units']
    assert cap_unit == 'Amp-hour', f'Expected capacity in Amp-hours, got {cap_unit} instead!'
    ce_unit = ce_info['units']
    assert len(ce_unit) == 0, f'Coulombic Efficiency extracted with units {ce_unit}; expected pure number!'
    cap_soc = max_cap_info['soc_level']
    assert len(cap_soc) == 0, 'Extracted capacity is a function of SOC!'
