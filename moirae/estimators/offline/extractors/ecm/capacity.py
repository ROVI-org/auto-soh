"""
Defines capacity extractor
"""
from typing import Dict, List, Tuple, TypedDict, Union
from warnings import warn

import pandas as pd
from scipy.integrate import trapezoid

from battdat.data import BatteryDataset
from battdat.postprocess.tagging import AddState
from battdat.schemas.column import ChargingState

from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset
from moirae.estimators.offline.DataCheckers.RPT import CapacityDataChecker
from moirae.estimators.offline.extractors.base import BaseExtractor, ExtractedParameter


class ValidSlowFullDoDSegment(TypedDict):
    """
    Auxiliary dictionary to contain information about valid segments that cover 100% depth-of-discharge (DoD) and are
    slow enough for capacity extraction.

    Args:
        cycled_charge: total amount of charge (in Amp-seconds) cycled during the segment (as observed by the cycler)
        duration: total duration (in seconds) of the segment
    """
    cycled_charge: float
    duration: float


class ChargeDischargeSegs(TypedDict):
    """
    Auxiliary class to store the valid charge and discharge segments of a given cycle.

    Args:
        charge: charge segment(s)
        discharge: discharge segment(s)
    """
    charge: Union[ValidSlowFullDoDSegment, List[ValidSlowFullDoDSegment]]
    discharge: Union[ValidSlowFullDoDSegment, List[ValidSlowFullDoDSegment]]


class MaxCapacityCoulEffExtractor(BaseExtractor):
    """
    Estimates the maximum discharge capacity of a battery from a low C-rate 100% DoD cycle

    Args:
        data_checker: data checker for capacity check tests
    """
    def __init__(self,
                 data_checker: CapacityDataChecker = CapacityDataChecker()) -> None:
        self.data_checker = data_checker

    @property
    def data_checker(self) -> CapacityDataChecker:
        return self._data_checker

    @data_checker.setter
    def data_checker(self, checker: CapacityDataChecker) -> None:
        if not isinstance(checker, CapacityDataChecker):
            raise TypeError('Data checker must be a CapacityDataChecker object!')
        self._data_checker = checker

    @property
    def voltage_limits(self) -> Union[Tuple[float, float], None]:
        """
        Voltage limits of the cell, in volts
        """
        return self._data_checker.voltage_limits

    @property
    def volt_tol(self) -> float:
        """
        Absolute voltage tolerance for voltage limits, in volts
        """
        return self._data_checker.voltage_tolerance

    @property
    def min_segment_duration(self) -> float:
        """
        Minimum duration (in seconds) for a charge or discharge segment to be considered valid
        """
        return 3600. / self._data_checker.max_C_rate

    def get_all_valid_segments(self, data: Union[pd.DataFrame, BatteryDataset]) -> ChargeDischargeSegs:
        """
        Function that returns all charge and discharge segments that meet the criteria established by the voltage limits
        and by the maximum C-rate. Assumes the data to have already been checked.

        Args:
            data: dataset that satisfied the data checker conditions

        Returns:
            dictionary of discharge and charge segments
        """
        # Ensure type
        data = ensure_battery_dataset(data=data)

        # Get raw data
        raw_data = data.tables.get('raw_data')
        if 'state' not in raw_data.columns:
            AddState().enhance(data=raw_data)

        # Find charge and discharge segments that spans voltage range
        charge_segments = raw_data[raw_data['state'] == ChargingState.charging]
        discharge_segments = raw_data[raw_data['state'] == ChargingState.discharging]

        # Aggregate the valid segments
        valid_segments = [[], []]
        for i, direction in enumerate([discharge_segments, charge_segments]):
            for (_, step_data) in direction.groupby('step_index'):
                include_step = False
                duration = step_data['test_time'].iloc[-1] - step_data['test_time'].iloc[0]
                if duration >= self.min_segment_duration:
                    # If the step lasts for long enough
                    if self.voltage_limits is not None:
                        # If we must check the voltage limits
                        min_volt, max_volt = sorted(self.voltage_limits)
                        min_obs, max_obs = step_data['volage'].min(), step_data['volage'].max()
                        if (abs(min_volt - min_obs) <= self.volt_tol) and (abs(max_volt - max_obs) <= self.volt_tol):
                            # If the voltage limits are reached
                            include_step = True
                    else:
                        include_step = True
                if include_step:
                    # Compute cycled capacity
                    cycled_charge = trapezoid(y=step_data['current'], x=step_data['test_time'])
                    valid_segments[i].append({'cycled_charge': abs(cycled_charge), 'duration': duration})

        # Assemble return dictionary
        info_dict = ChargeDischargeSegs(discharge=valid_segments[0],
                                        charge=valid_segments[1])

        return info_dict

    def get_best_valid_segments(self, data: Union[pd.DataFrame, BatteryDataset]) -> ChargeDischargeSegs:
        """
        Function that returns longest charge and discharge segments that meet the criteria established by the voltage
        limits and by the maximum C-rate. Assumes the data to have already been checked.

        Args:
            data: dataset that satisfied the data checker conditions

        Returns:
            dictionary of discharge and charge segments
        """
        # Ensure type
        data = ensure_battery_dataset(data=data)

        # Get all the valid segments
        segments = self.get_all_valid_segments(data=data)

        # Get the longest of each segment
        longest_segs = {}
        for state, segs in segments.items():
            max_dur = -1.  # All durations are positive, so any negative number for initialization is fine
            best_segment = None
            for seg_info in segs:
                if seg_info['duration'] > max_dur:
                    max_dur = seg_info['duration']
                    best_segment = seg_info.copy()
            longest_segs[state] = best_segment

        return longest_segs

    def compute_parameters(self,
                           data: Union[pd.DataFrame, BatteryDataset]) -> Tuple[ExtractedParameter, ExtractedParameter]:
        """
        Computes the maximum capacity and Coulombic efficiency from the given data, assuming it has already passed the
        necessary checks

        Args:
            data: data to be used

        Returns:
            maximum discharge capacity and Coulombic efficiency
        """
        # Ensure type
        data = ensure_battery_dataset(data=data)

        # Get the longest of each charge and discharge segments
        segments = self.get_best_valid_segments(data=data)
        dis_segment = segments['discharge']
        chg_segment = segments['charge']

        # Now, we can compute the maximum discharge capacity, as well as the Coulombic efficiency
        q_t = dis_segment['cycled_charge'] / 3600.  # Convert to Amp-hours
        ce = dis_segment['cycled_charge'] / chg_segment['cycled_charge']

        if ce > 1.:
            warn(f'Computed Coulombic Efficiency of {100 * ce:1.1f} % > 100%! '
                 'Please double check the data, including C-rates for different segments.')

        # Prepare return objects
        capacity = ExtractedParameter(value=q_t,
                                      units='Amp-hour',
                                      soc_level=[])
        coul_eff = ExtractedParameter(value=ce,
                                      units='',
                                      soc_level=[])
        return capacity, coul_eff

    def extract(self, data: Union[pd.DataFrame, BatteryDataset]) -> Tuple[ExtractedParameter, ExtractedParameter]:
        # Check data
        data = self._data_checker.check(data=data)
        
        # Compute what is needed
        return self.compute_parameters(data=data)
