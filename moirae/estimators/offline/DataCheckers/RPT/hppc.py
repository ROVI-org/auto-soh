from typing import TypedDict, Union
from typing_extensions import NotRequired, Self

import pandas as pd

from battdat.data import BatteryDataset
from battdat.postprocess.tagging import AddState, AddSteps, AddMethod, AddSubSteps
from battdat.schemas.column import ChargingState, ControlMethod

from moirae.models.ecm.components import MaxTheoreticalCapacity
from moirae.estimators.offline.DataCheckers import DeltaSOCRangeChecker, DataCheckError
from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset


class FullHPPCCheckerPreinitParams(TypedDict):
    """
    Collection of parameters to be used when initialize the FullHPPCDataChecker once capacity and coulombic efficiency
    are obtained.
    """
    min_delta_soc: NotRequired[float]
    min_pulses: NotRequired[int]
    ensure_bidirectional: NotRequired[bool]
    min_number_of_rests: NotRequired[int]
    min_rest_duration: NotRequired[float]
    min_rest_prev_dur: NotRequired[float]
    rest_current_threshold: NotRequired[float]

    @classmethod
    def default(cls) -> Self:
        return {'min_delta_soc': 0.1,
                'min_pulses': 1,
                'ensure_bidirectional': True,
                'min_number_of_rests': True,
                'min_rest_duration': 600,
                'min_rest_prev_dur': 300.,
                'rest_current_threshold': 1.0e-04}


class PulseDataChecker(DeltaSOCRangeChecker):
    """
    Ensures the cycle provided is representative of a Hybrid Pulse Power Characterization (HPPC) diagnostic cycle, that
    is, it contains a sufficient number of pulses and covers a sufficient SOC range.

    Args:
        capacity: Assumed cell capacity in Amp-hours
        coulombic_efficiency: Assumed Coulombic efficiency of the cell; defaults to 1.0 (100%)
        min_delta_soc: Minimum required SOC change; defaults to 10%
        min_pulses: Minimum number of pulses (both charge and discharge) required for the cycle to be considered a HPPC
        ensure_bidirectional: If True, ensures that both charge and discharge pulses are present
    """
    def __init__(self,
                 capacity: Union[float, MaxTheoreticalCapacity],
                 coulombic_efficiency: float = 1.0,
                 min_delta_soc: float = 0.1,
                 min_pulses: int = 1,
                 ensure_bidirectional: bool = True):
        super().__init__(capacity=capacity, coulombic_efficiency=coulombic_efficiency, min_delta_soc=min_delta_soc)
        self.min_pulses = min_pulses
        self.ensure_bidirectional = ensure_bidirectional

    def check(self,
              data: Union[pd.DataFrame, BatteryDataset]) -> BatteryDataset:

        # Make sure we have a single cycle and the SOC range is sufficient
        data = super().check(data=data)

        # Get only raw data
        raw_data = data.tables.get('raw_data')

        # Ensure we have the necessary state, step, and method columns
        if 'state' not in raw_data.columns:
            AddState().enhance(raw_data)
        if 'step_index' not in raw_data.columns:
            AddSteps().enhance(raw_data)
        if 'method' not in raw_data.columns:
            AddMethod().enhance(raw_data)
        if 'substep_index' not in raw_data.columns:
            AddSubSteps().enhance(raw_data)

        # Now, make sure we can find at least `min_pulses`
        pulses = raw_data[raw_data['method'] == ControlMethod.pulse]
        num_observed_pulses = len(pulses['substep_index'].unique())
        if num_observed_pulses < self.min_pulses:
            raise DataCheckError(f"Cycle contains only {num_observed_pulses} pulses; "
                                 f"expected at least {self.min_pulses}!")

        # Check if we have bidirectional pulses if required
        if self.ensure_bidirectional:
            charge_pulses = pulses[pulses['state'] == ChargingState.charging]
            discharge_pulses = pulses[pulses['state'] == ChargingState.discharging]
            num_charge_pulses = len(charge_pulses['substep_index'].unique())
            num_discharge_pulses = len(discharge_pulses['substep_index'].unique())
            if num_charge_pulses == 0:
                raise DataCheckError("No charge pulses found in the cycle!")
            if num_discharge_pulses == 0:
                raise DataCheckError("No discharge pulses found in the cycle!")
            # Make sure they don't differ by more than 1, which only happens if we are close to a voltage limit
            if abs(num_charge_pulses - num_discharge_pulses) > 1:
                raise DataCheckError(f"Found {num_charge_pulses} charge and {num_discharge_pulses} discharge pulses; "
                                     f"HPPC is not bi-directional!")

        # Include these modifications back in the data
        data.tables['raw_data'] = raw_data

        return data


class RestDataChecker(DeltaSOCRangeChecker):
    """
    Ensures the cycle provided contains rest period(s).

    Args:
        capacity: Assumed cell capacity in Amp-hours
        coulombic_efficiency: Assumed Coulombic efficiency of the cell; defaults to 1.0 (100%)
        min_delta_soc: Minimum required SOC change; defaults to 10%
        min_rest_duration: Minimum duration of rest period in seconds; defaults to 600s (10 min)
        min_prev_duration: Minimum duration of the step that precedes a rest; defaults to 300s (5 min)
        min_number_of_rests: Minimum number of rest periods required; defaults to 1
        rest_current_threshold: Maximum absolute current to consider as rest; defaults to 1 mA
    """
    def __init__(self,
                 capacity: Union[float, MaxTheoreticalCapacity],
                 coulombic_efficiency: float = 1.0,
                 min_delta_soc: float = 0.1,
                 min_number_of_rests: int = 1,
                 min_rest_duration: float = 600.,
                 min_prev_duration: float = 300.,
                 rest_current_threshold: float = 1.0e-04):
        super().__init__(capacity=capacity, coulombic_efficiency=coulombic_efficiency, min_delta_soc=min_delta_soc)
        self.min_rests = min_number_of_rests
        self.min_rest_duration = min_rest_duration
        self.min_prev_duration = min_prev_duration
        self.rest_current_threshold = rest_current_threshold

    @property
    def min_rests(self) -> int:
        """Minimum number of rest periods required"""
        return self._min_rests

    @min_rests.setter
    def min_rests(self, value: int):
        if value < 1:
            raise ValueError("Minimum number of rests must be at least 1!")
        self._min_rests = value

    @property
    def min_rest_duration(self) -> float:
        """Minimum duration of rest period in seconds"""
        return self._min_rest_dur

    @min_rest_duration.setter
    def min_rest_duration(self, value: float):
        if value <= 0:
            raise ValueError("Minimum rest duration must be positive!")
        self._min_rest_dur = value

    @property
    def min_prev_duration(self) -> float:
        return self._min_prev_dur

    @min_prev_duration.setter
    def min_prev_duration(self, value: float):
        if value <= 0:
            raise ValueError('Minimum duration of step that precedes rest must be positive!')
        self._min_prev_dur = value

    @property
    def rest_current_threshold(self) -> float:
        """Maximum absolute current to consider as rest"""
        return self._curr_thresh

    @rest_current_threshold.setter
    def rest_current_threshold(self, value: float):
        self._curr_thresh = abs(value)

    def check(self,
              data: Union[pd.DataFrame, BatteryDataset]) -> BatteryDataset:
        # Make sure we have a single cycle and the SOC range is sufficient
        data = super().check(data=data)

        # Get only raw data
        raw_data = data.tables.get('raw_data')

        # Ensure we have the necessary state, step, and method columns
        if 'state' not in raw_data.columns:
            AddState(rest_curr_threshold=self.rest_current_threshold).enhance(raw_data)
        if 'step_index' not in raw_data.columns:
            AddSteps().enhance(raw_data)

        # Now, let's find the rest periods
        rest_data = raw_data[raw_data['state'] == ChargingState.rest]
        rest_periods = []
        for rest_step, rest in rest_data.groupby('step_index'):
            rest_dur = rest['test_time'].iloc[-1] - rest['test_time'].iloc[0]
            if rest_dur < self._min_rest_dur:
                continue
            # Now, get the previous step
            prev_step = raw_data[raw_data['step_index'] == rest_step - 1]
            if len(prev_step) == 0:  # If the first rest is at the very beginning of the cycle
                continue
            prev_dur = prev_step['test_time'].iloc[-1] - prev_step['test_time'].iloc[0]
            if prev_dur >= self._min_prev_dur:
                rest_periods.append(rest_step)

        if len(rest_periods) < self.min_rests:
            raise DataCheckError(f"Cycle contains only {len(rest_periods)} rest periods of at least "
                                 f"{self.min_rest_duration:.1f} seconds with previous steps lasting at "
                                 f"least {self._min_prev_dur:.1f} seconds; "
                                 f"expected at least {self.min_rests:d} rest periods!")

        # Include these modifications back in the data
        data.tables['raw_data'] = raw_data

        return data


class FullHPPCDataChecker():
    """
    Ensures the cycle provided contains both pulses and rest periods

    Args:
        capacity: Assumed cell capacity in Amp-hours
        coulombic_efficiency: Assumed Coulombic efficiency of the cell; defaults to 1.0 (100%)
        min_delta_soc: Minimum required SOC change; defaults to 10%
        min_pulses: Minimum number of pulses (both charge and discharge) required for the cycle to be considered a HPPC
        ensure_bidirectional: If True, ensures that both charge and discharge pulses are present
        min_rest_duration: Minimum duration of rest period in seconds; defaults to 600s (10 min)
        min_number_of_rests: Minimum number of rest periods required; defaults to 1
        min_rest_prev_dur: Mimimum duration of steps the precede rests; defaults to 300s (5 min)
        rest_current_threshold: Maximum absolute current to consider as rest; defaults to 1 mA
    """
    def __init__(self,
                 capacity: Union[float, MaxTheoreticalCapacity],
                 coulombic_efficiency: float = 1.0,
                 min_delta_soc: float = 0.1,
                 min_pulses: int = 1,
                 ensure_bidirectional: bool = True,
                 min_number_of_rests: int = 1,
                 min_rest_duration: float = 600.,
                 min_rest_prev_dur: float = 300.,
                 rest_current_threshold: float = 1.0e-04
                 ):
        self.pulse_checker = PulseDataChecker(capacity=capacity,
                                              coulombic_efficiency=coulombic_efficiency,
                                              min_delta_soc=min_delta_soc,
                                              min_pulses=min_pulses,
                                              ensure_bidirectional=ensure_bidirectional)
        self.rest_checker = RestDataChecker(capacity=capacity,
                                            coulombic_efficiency=coulombic_efficiency,
                                            min_delta_soc=min_delta_soc,
                                            min_number_of_rests=min_number_of_rests,
                                            min_rest_duration=min_rest_duration,
                                            min_prev_duration=min_rest_prev_dur,
                                            rest_current_threshold=rest_current_threshold)

    @property
    def capacity(self) -> float:
        return self.pulse_checker.capacity

    @capacity.setter
    def capacity(self, value: Union[float, MaxTheoreticalCapacity]):
        self.pulse_checker.capacity = value
        self.rest_checker.capacity = value

    @property
    def coulombic_efficiency(self) -> float:
        return self.rest_checker.coulombic_efficiency

    @coulombic_efficiency.setter
    def coulombic_efficiency(self, value: float):
        self.pulse_checker.coulombic_efficiency = value
        self.rest_checker.coulombic_efficiency = value

    @property
    def min_delta_soc(self) -> float:
        return self.rest_checker.min_delta_soc

    @min_delta_soc.setter
    def min_delta_soc(self, value: float):
        self.pulse_checker.min_delta_soc = value
        self.rest_checker.min_delta_soc = value

    def check(self, data: Union[pd.DataFrame, BatteryDataset]) -> BatteryDataset:
        # Ensure we have a BatteryDataset
        data = ensure_battery_dataset(data)

        # Get pulses
        data = self.pulse_checker.check(data=data)

        # Get rests
        data = self.rest_checker.check(data=data)

        return data
