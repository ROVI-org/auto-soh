from pytest import raises, fixture

from moirae.estimators.offline.DataCheckers import DeltaSOCRangeChecker
from moirae.estimators.offline.DataCheckers import DataCheckError


@fixture
def soc_checker(simple_rint):
    capacity = simple_rint[0].q_t
    return DeltaSOCRangeChecker(capacity=capacity, min_delta_soc=0.95)


def test_delta_soc_checker_init_errors():
    with raises(ValueError, match="Minimum SOC change must be in the range \[0, 1\]."):
        DeltaSOCRangeChecker(capacity=1., min_delta_soc=-0.1)
    with raises(ValueError, match="Minimum SOC change must be in the range \[0, 1\]."):
        DeltaSOCRangeChecker(capacity=1., min_delta_soc=1.1)
    with raises(ValueError, match="Capacity must be a positive number!"):
        DeltaSOCRangeChecker(capacity=-1.0)
    with raises(TypeError, match="Capacity must be a float or MaxTheoreticalCapacity object!"):
        DeltaSOCRangeChecker(capacity="invalid")


def test_delta_soc_checker_incomplete(timeseries_dataset, soc_checker):
    # Create incomplete dataset
    raw = timeseries_dataset.tables['raw_data']
    incomplete = raw[raw['test_time'] <= 905]

    with raises(DataCheckError, match="Dataset must sample at least 95.0% of SOC. Only sampled 50.0%."):
        soc_checker.check(data=incomplete)


def test_delta_soc_checker(realistic_rpt_data, soc_checker):
    # Get raw data
    raw_rpt = realistic_rpt_data.tables['raw_data']
    cap_check = raw_rpt[raw_rpt['protocol'] == b'Capacity Check']

    # Should pass on single cycle data
    soc_checker.check(data=cap_check)
