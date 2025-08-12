from pytest import raises

from battdat.postprocess.tagging import AddState, AddSteps, AddMethod, AddSubSteps
from battdat.postprocess.integral import StateOfCharge
from battdat.schemas.column import ChargingState

from moirae.estimators.offline.DataCheckers import DataCheckError
from moirae.estimators.offline.DataCheckers.RPT import RestDataChecker


def test_init_errors() -> None:
    """
    Test that the RestDataChecker raises errors for invalid initialization parameters
    """

    # Test negative minimum number of rests
    with raises(ValueError, match="Minimum number of rests must be at least 1!"):
        RestDataChecker(capacity=10., min_number_of_rests=-1)
    with raises(ValueError, match="Minimum number of rests must be at least 1!"):
        RestDataChecker(capacity=10., min_number_of_rests=0)

    # Test negative minimum rest duration
    with raises(ValueError, match="Minimum rest duration must be positive!"):
        RestDataChecker(capacity=10., min_rest_duration=-10.)

    with raises(ValueError, match="Minimum rest duration must be positive!"):
        RestDataChecker(capacity=10., min_rest_duration=0.)

    # Make sure current threshold is positive
    rest_checker = RestDataChecker(capacity=10., rest_current_threshold=-1.)
    assert rest_checker.rest_current_threshold == 1.0, "Current threshold should be converted to positive automatically"


def test_unsatisfactory_rests(realistic_rpt_data, realistic_LFP_aSOH) -> None:
    """
    Test that the RestDataChecker raises an error if the number of rests is below the minimum
    """
    # Create unreasonable checkers expecting 300 rests, or expecting very long rests
    checke_many_rests = RestDataChecker(capacity=realistic_LFP_aSOH.q_t,
                                        min_delta_soc=0.1,
                                        min_number_of_rests=300)
    check_long_rests = RestDataChecker(capacity=realistic_LFP_aSOH.q_t,
                                        min_delta_soc=0.1,
                                        min_rest_duration=3600 * 10.)

    # Get HPPC data
    raw_rpt = realistic_rpt_data.tables['raw_data']
    hppc = raw_rpt[raw_rpt['protocol'] == b'Full HPPC']

    # Start with many rests
    message = "Cycle contains only 12 rest periods of at least 600.0 seconds; expected at least 300!"
    with raises(DataCheckError, match=message):
        checke_many_rests.check(data=hppc)
    
    # Now, check long rests
    message = "Cycle contains only 0 rest periods of at least 36000.0 seconds; expected at least 1!"
    with raises(DataCheckError, match=message):
        check_long_rests.check(data=hppc)


def test_satisfactory_rests(realistic_rpt_data, realistic_LFP_aSOH) -> None:
    """
    Test that the RestDataChecker does not raise an error if the number of rests is above the minimum
    """
    # Create reasonable checkers expecting 1 rest, or expecting 10 min long rests
    checker = RestDataChecker(capacity=realistic_LFP_aSOH.q_t,
                              min_delta_soc=0.99,
                              min_number_of_rests=10,
                              min_rest_duration=1800.,
                              rest_current_threshold=1.0e-04)

    # Get HPPC data
    raw_rpt = realistic_rpt_data.tables['raw_data']
    hppc = raw_rpt[raw_rpt['protocol'] == b'Full HPPC']

    assert len(checker.check(data=hppc, extract=True)) == 12, "Should find 12 valide rests in the cycle"