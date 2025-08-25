import numpy as np
from pytest import raises

from moirae.estimators.offline.extractors.ecm import OCVExtractor
from moirae.estimators.offline.assemblers.ecm import OCVAssembler


def test_wrong_units():
    ocv_ass = OCVAssembler()
    with raises(ValueError, match='OCV provided in Volts, rather than Volt!'):
        _ = ocv_ass.assemble(extracted_parameter={'units': 'Volts',
                                                  'value': [0., 1.],
                                                  'soc_level': [0., 1.],
                                                  'current': [1., 1.]})


def test_proper_assembly(realistic_rpt_data, realistic_LFP_aSOH):
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

    # Assemble new OCV
    gt_soc_pts = ocv_gt.ocv_ref.soc_pinpoints.flatten()
    ocv_assembler = OCVAssembler(soc_points=gt_soc_pts)
    ocv_ass = ocv_assembler.assemble(extracted_parameter=ocv_info)
    soc_test = np.linspace(0., 1., 100)
    assert np.allclose(ocv_ass(soc_test), ocv_gt(soc_test), rtol=1.0e-02), \
        'More than 1% discrepancy in OCV assembled with exact SOC pinpoints!'
    errors = ocv_ass(soc_test) - ocv_gt(soc_test)
    mae = np.mean(abs(errors))
    rmse = np.sqrt(np.mean(np.pow(errors, 2)))
    assert mae <= 5.0e-03, f'MAE = {mae:.1e} V > 5 mV!'
    assert rmse <= 7.5e-03, f'RMSE = {rmse:.1e} V > 7.5 mV!'
    # Change the SOC points
    new_soc = np.linspace(0., 1., 11)  # one every 10%
    ocv_assembler.soc_points = new_soc
    ocv_ass = ocv_assembler.assemble(extracted_parameter=ocv_info)
    assert np.allclose(ocv_ass(new_soc), ocv_gt(new_soc), rtol=1.0e-02), \
        'More than 1% discrepancy in OCV assembled with different SOC pinpoints!'
    errors = ocv_ass(soc_test) - ocv_gt(soc_test)
    mae = np.mean(abs(errors))
    rmse = np.sqrt(np.mean(np.pow(errors, 2)))
    assert mae <= 2.5e-02, f'MAE = {mae:.1e} V > 25 mV!'
    assert rmse <= 5.5e-02, f'RMSE = {rmse:.1e} V > 55 mV!'
