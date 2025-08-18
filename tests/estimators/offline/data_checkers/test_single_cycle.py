from pytest import raises

from moirae.estimators.offline.DataCheckers import SingleCycleChecker
from moirae.estimators.offline.DataCheckers import DataCheckError


def test_single_cycle_checker(realistic_rpt_data):
    # Get RPT Dataset
    checker = SingleCycleChecker()

    # Get raw data
    raw_rpt = realistic_rpt_data.tables['raw_data']
    hppc_data = raw_rpt[raw_rpt['protocol'] == 'Full HPPC']
    cap_check = raw_rpt[raw_rpt['protocol'] == 'Capacity Check']

    # Should pass on single cycle data
    checker.check(data=hppc_data)
    checker.check(data=cap_check)

    # Should fail on both
    with raises(DataCheckError, match="Multiple cycles found in data! Please provide a single cycle."):
        checker.check(data=realistic_rpt_data)
