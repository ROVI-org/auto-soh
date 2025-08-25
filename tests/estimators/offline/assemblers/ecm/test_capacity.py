from pytest import raises

from moirae.estimators.offline.assemblers.ecm import CapacityAssembler


def test_wrong_params():
    cap_ass = CapacityAssembler()
    with raises(ValueError, match='Capacity provided in Amp, rather than Amp-hour'):
        cap_ass.assemble(extracted_parameter={'units': 'Amp',
                                              'value': 30.,
                                              'soc_level': []})
