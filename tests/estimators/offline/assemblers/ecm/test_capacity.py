from pytest import raises

from moirae.estimators.offline.assemblers.ecm import CapacityAssembler


def test_wrong_params():
    cap_ass = CapacityAssembler()
    with raises(ValueError, match='Capacity provided in Amp, rather than Amp-hour'):
        cap_ass.assemble(extracted_parameter={'units': 'Amp',
                                              'value': 30.,
                                              'soc_level': []})

    with raises(ValueError, match='Negative capacity -30.00 provided!'):
        cap_ass.assemble(extracted_parameter={'units': 'Amp-hour',
                                              'value': -30.,
                                              'soc_level': []})


def test_correct_assembly():
    cap_ass = CapacityAssembler()
    qt = cap_ass.assemble(extracted_parameter={'units': 'Amp-hour',
                                               'value': 30.,
                                               'soc_level': []})
    assert qt.amp_hour.item() == 30., 'Wrong capacity assembled!'
