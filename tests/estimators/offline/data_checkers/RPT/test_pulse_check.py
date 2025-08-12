from pytest import raises

from battdat.postprocess.tagging import AddState, AddSteps, AddMethod, AddSubSteps
from battdat.postprocess.integral import StateOfCharge
from battdat.schemas.column import ChargingState

from moirae.estimators.offline.DataCheckers import DataCheckError
from moirae.estimators.offline.DataCheckers.RPT import PulseDataChecker


def test_hppc_min_pulses(realistic_rpt_data, realistic_LFP_aSOH) -> None:
    """Test that the HPPCDataChecker raises an error if the number of pulses is below the minimum"""
    # Create unreasonable checker expecting 300 pulses
    checker = PulseDataChecker(capacity=realistic_LFP_aSOH.q_t,
                               min_delta_soc=0.1,
                               min_pulses=300)

    # Get HPPC data
    raw_rpt = realistic_rpt_data.tables['raw_data']
    hppc = raw_rpt[raw_rpt['protocol'] == b'Full HPPC']

    with raises(ValueError, match="Cycle contains only 20 pulses; expected at least 300!"):
        checker.check(data=hppc)


def test_hppc_bidirectional_pulses(realistic_rpt_data, realistic_LFP_aSOH) -> None:
    """Test that the HPPCDataChecker raises an error if bidirectional pulses are not present"""
    # Create unreasonable checker expecting at least one charge and one discharge pulse
    checker = PulseDataChecker(capacity=realistic_LFP_aSOH.q_t,
                               min_delta_soc=0.1,
                               min_pulses=1,
                               ensure_bidirectional=True)

    # Process raw data
    raw_rpt = realistic_rpt_data.tables['raw_data']
    AddState().enhance(raw_rpt)
    AddSteps().enhance(raw_rpt)
    AddMethod().enhance(raw_rpt)
    AddSubSteps().enhance(raw_rpt)
    StateOfCharge().enhance(raw_rpt)

    # Get HPPC data
    hppc = raw_rpt[raw_rpt['protocol'] == b'Full HPPC']

    with raises(DataCheckError, match="No charge pulses found in the cycle!"):
        checker.check(data=hppc[hppc['state'] == ChargingState.discharging])

    with raises(DataCheckError, match="No discharge pulses found in the cycle!"):
        checker.check(data=hppc[hppc['state'] == ChargingState.charging])

    with raises(DataCheckError, match="Found 1 charge and 3 discharge pulses; HPPC is not bi-directional!"):
        checker.check(data=hppc[hppc['substep_index'] <= 14])


def test_hppc_checker(realistic_rpt_data, realistic_LFP_aSOH) -> None:
    """Test the HPPCDataChecker with a valid HPPC cycle"""
    # Create a checker with reasonable parameters
    checker = PulseDataChecker(capacity=realistic_LFP_aSOH.q_t,
                               min_delta_soc=0.99,
                                 min_pulses=20,
                                 ensure_bidirectional=True)

    # Get HPPC data
    raw_rpt = realistic_rpt_data.tables['raw_data']
    hppc = raw_rpt[raw_rpt['protocol'] == b'Full HPPC']

    # Check that it passes without errors
    checker.check(data=hppc)
