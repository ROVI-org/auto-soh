import numpy as np
from pytest import raises

from moirae.estimators.offline.extractors.ecm import RCExtractor
from moirae.estimators.offline.assemblers.utils import SOCRegressor
from moirae.estimators.offline.assemblers.ecm import CapacitanceAssembler


def test_wrong_unit():
    r0_assember = CapacitanceAssembler()
    with raises(ValueError, match="Resistance provided in Ohm, rather than Farad!"):
        _ = r0_assember.assemble(extracted_parameter={'units': 'Ohm',
                                                      'value': [1., 1.5],
                                                      'soc_level': [0., 1.]})


def test_neg_vals():
    regressor = SOCRegressor(style='smooth')
    r_ass_interp = CapacitanceAssembler(regressor=regressor)
    with raises(ValueError, match="Non-positive capacitance detected! Base values = "):
        r_assembled = r_ass_interp.assemble(extracted_parameter={'value': -np.ones(11),
                                                                 'soc_level': np.linspace(0, 1, 11),
                                                                 'units': 'Farad'})


def test_proper_assembly(realistic_rpt_data, realistic_LFP_aSOH):
    # Get the relevant part of the data
    raw_rpt = realistic_rpt_data.tables.get('raw_data')
    hppc = raw_rpt[raw_rpt['protocol'] == 'Full HPPC']

    # Prepare what is needed
    capacity = realistic_LFP_aSOH.q_t
    coul_eff = realistic_LFP_aSOH.ce
    capacitance_gt = realistic_LFP_aSOH.rc_elements[0].c

    # Extract RC
    rc_extractor = RCExtractor.init_from_basics(capacity=capacity,
                                                coulombic_efficiency=coul_eff,
                                                min_delta_soc=0.99,
                                                min_number_of_rests=10)
    rc_info = rc_extractor.extract(data=hppc, start_soc=hppc['SOC'].iloc[0])
    capacitance_info = rc_info[0][1]

    # Prepare and test different assembler
    soc_test = np.linspace(0., 1., 100)

    # Interpolation equivalent to interp1d# Interpolation equivalent to interp1d
    cap_assembler = CapacitanceAssembler(regressor=SOCRegressor(parameters={'k': 1}))
    cap_ass = cap_assembler.assemble(extracted_parameter=capacitance_info)
    assert np.allclose(cap_ass.get_value(soc=soc_test), capacitance_gt.get_value(soc=soc_test), rtol=0.05), \
        'Larger than 5% different on liner-interpolation-based assembly!'

    # LSQUnivariate
    regressor = SOCRegressor(style='lsq', parameters={'k': 1})
    soc_pts = [-0.05] + np.linspace(0., 1., 6).tolist() + [1.05]
    cap_assembler = CapacitanceAssembler(regressor=regressor,
                                       soc_points=soc_pts)
    c_ass = cap_assembler.assemble(extracted_parameter=capacitance_info)
    assert np.allclose(c_ass.get_value(soc=soc_test), capacitance_gt.get_value(soc=soc_test), rtol=0.05), \
        'Larger than 5% different on LSQ-based assembly!'

    # Smoothing spline
    regressor = SOCRegressor(style='smooth')
    cap_assembler = CapacitanceAssembler(regressor=regressor)
    c_ass = cap_assembler.assemble(extracted_parameter=capacitance_info.copy())
    assert np.allclose(c_ass.get_value(soc=soc_test), capacitance_gt.get_value(soc=soc_test), rtol=0.05), \
        f'Larger than 5% different on spline-based assembly!'
