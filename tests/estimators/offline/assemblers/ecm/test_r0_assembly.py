import numpy as np
from pytest import raises

from moirae.estimators.offline.extractors.ecm import R0Extractor
from moirae.estimators.offline.assemblers.utils import SOCRegressor
from moirae.estimators.offline.assemblers.ecm import ResistanceAssembler


def test_wrong_unit():
    r0_assember = ResistanceAssembler()
    with raises(ValueError, match="Resistance provided in Volt, rather than Ohm!"):
        _ = r0_assember.assemble(extracted_parameter={'units': 'Volt',
                                                      'value': [1., 1.5],
                                                      'soc_level': [0., 1.]})


def test_proper_assembly(realistic_rpt_data, realistic_LFP_aSOH):
    # Get the relevant part of the data
    raw_rpt = realistic_rpt_data.tables.get('raw_data')
    hppc = raw_rpt[raw_rpt['protocol'] == 'Full HPPC']

    # Prepare what is needed
    capacity = realistic_LFP_aSOH.q_t
    coul_eff = realistic_LFP_aSOH.ce
    r0_gt = realistic_LFP_aSOH.r0
    r0_extractor = R0Extractor.init_from_basics(capacity=capacity,
                                                coulombic_efficiency=coul_eff,
                                                min_delta_soc=0.99,
                                                min_pulses=10,
                                                ensure_bidirectional=True,
                                                min_number_of_rests=10)

    # Recall we start at close to 100% SOC
    start_soc = hppc['SOC'].iloc[0]

    # Extract
    r0_info = r0_extractor.extract(data=hppc, start_soc=start_soc)

    # Prepare and test different assembler
    soc_test = np.linspace(0., 1., 100)

    # Interpolation equivalent to interp1d
    r_ass_interp = ResistanceAssembler(regressor=SOCRegressor(parameters={'k': 1}))
    r_assembled = r_ass_interp.assemble(extracted_parameter=r0_info.copy())
    assert np.allclose(r_assembled.get_value(soc=soc_test), r0_gt.get_value(soc=soc_test), rtol=0.05), \
        'Larger than 5% different on liner-interpolation-based assembly!'

    # LSQUnivariate
    regressor = SOCRegressor(style='lsq', parameters={'k': 1})
    soc_pts = [-0.05] + np.linspace(0., 1., 11).tolist() + [1.05]
    r_ass_interp = ResistanceAssembler(regressor=regressor,
                                       soc_points=soc_pts)
    r_assembled = r_ass_interp.assemble(extracted_parameter=r0_info.copy())
    assert np.allclose(r_assembled.get_value(soc=soc_test), r0_gt.get_value(soc=soc_test), rtol=0.05), \
        'Larger than 5% different on LSQ-based assembly!'

    # Smoothing spline
    regressor = SOCRegressor(style='smooth', parameters={'lam': 6.0e-03})
    r_ass_interp = ResistanceAssembler(regressor=regressor)
    r_assembled = r_ass_interp.assemble(extracted_parameter=r0_info.copy())
    assert np.allclose(r_assembled.get_value(soc=soc_test), r0_gt.get_value(soc=soc_test), rtol=0.05), \
        f'Larger than 5% different on spline-based assembly!'

    # Now, test one that fails due to negative values
    regressor = SOCRegressor(style='smooth')
    r_ass_interp = ResistanceAssembler(regressor=regressor)
    with raises(ValueError, match="Non-positive resistance detected! Base values = "):
        r_assembled = r_ass_interp.assemble(extracted_parameter={'value': -np.ones(11),
                                                                 'soc_level': np.linspace(0, 1, 11),
                                                                 'units': 'Ohm'})
