"""
Unit test for the ECM data checkers.
"""
import pandas as pd

from pytest import fixture, raises

from moirae.estimators.offline.DataCheckers.base import DataCheckError
from moirae.estimators.offline.DataCheckers.RPT import CapacityDataChecker


@fixture
def capacity_checker() -> CapacityDataChecker:
    """
    Create a CapacityDataChecker instance with default parameters.
    """
    return CapacityDataChecker(voltage_limits=(2.5, 4.0), max_C_rate=1.0)


@fixture
def full_cap_check() -> pd.DataFrame:
    """
    Create a DataFrame representing a full capacity check cycle, but with a few misdirections, including segments
    that do not reach voltage limits, as well as segments that are too quick.
    """
    voltage = [2.5, 2.5, 2.9, 3.0, 3.2, 3.2, 3.2, 3.0, 2.9, 2.5, 2.5, 2.5, 2.9, 3.5, 3.8, 4.0, 4.0, 3.8, 3.5, 3.0, 2.5]
    current = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, -1., -1., -1., 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1., -1., -1., -1.]
    testime = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10., 11., 12., 15., 20., 21., 22., 23., 28., 30., 31.]

    slow_voltage = [2.5, 2.5, 2.5, 3.0, 3.5, 3.8, 4.0, 4.0, 4.0, 4.0, 3.8, 3.5, 3.0, 2.5, 2.5, 2.5]
    slow_current = [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, -.1, -.1, -.1, -.1, -.1, 0.0, -0.0]
    slow_testime = [32., 33., 34., 1000, 2000, 3000, 4000, 4100, 4200, 4300, 5300, 6300, 7300, 8300, 8400, 8500]

    # Create DataFrame
    full_volt = voltage + slow_voltage
    full_curr = current + slow_current
    full_time = testime + slow_testime
    cycle = [1] * len(full_volt)

    return pd.DataFrame.from_dict({'voltage': full_volt,
                                   'current': full_curr,
                                   'test_time': full_time,
                                   'cycle_number': cycle})


def test_capacity_data_checker(capacity_checker, full_cap_check) -> None:
    """
    Test the CapacityDataChecker with a full capacity check cycle.
    It should pass without raising any exceptions.
    """
    # Test the voltage limits
    misses_upper = full_cap_check.iloc[:10]
    with raises(DataCheckError, match="Cycle does not reach upper voltage limit of 4.00 V!"):
        capacity_checker.check(misses_upper)
    misses_lower = full_cap_check.iloc[12:19]
    with raises(DataCheckError, match="Cycle does not reach lower voltage limit of 2.50 V!"):
        capacity_checker.check(misses_lower)

    # Test duration
    short_dur = full_cap_check.iloc[:20]
    with raises(DataCheckError, match="Cycle does not contain a valid charge segment at 1.0C or lower!"):
        capacity_checker.check(short_dur)

    # Make sure that it passes with the full data
    capacity_checker.check(full_cap_check)


def test_cap_check_realistic(realistic_rpt_data) -> None:
    """
    Test the capacity checker with a realistic (though simulated) data
    """
    # Get relevant cycles
    raw_rpt = realistic_rpt_data.tables.get('raw_data')
    cap_check = raw_rpt[raw_rpt['protocol'] == 'Capacity Check']

    # Make sure it works fine
    CapacityDataChecker().check(data=cap_check)
