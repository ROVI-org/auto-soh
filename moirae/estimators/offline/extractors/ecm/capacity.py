"""
Defines capacity extractor
"""
from typing import Tuple, Union
from warnings import warn

import pandas as pd
from scipy.integrate import trapezoid

from battdat.data import BatteryDataset
from battdat.schemas.column import ChargingState

from moirae.estimators.offline.DataCheckers.RPT import CapacityDataChecker
from moirae.estimators.offline.extractors.base import BaseExtractor, ExtractedParameter


class MaxCapacityCoulEffExtractor(BaseExtractor):
    """
    Estimates the maximum discharge capacity of a battery from a low C-rate 100% DoD cycle

    Args:
        data_checker: data checker for capacity check tests
    """
    def __init__(self,
                 data_checker: CapacityDataChecker = CapacityDataChecker()) -> None:
        self._data_checker = data_checker

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

    def extract(self, data: Union[pd.DataFrame, BatteryDataset]) -> Tuple[ExtractedParameter, ExtractedParameter]:
        # Check data
        data = self._data_checker.check(data=data)
        raw_data = data.tables.get('raw_data')

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

        # Sort by longest duration
        for i in range(2):
            valid_segments[i].sort(key=lambda x: x['duration'],
                                   reverse=True)

        # Now, we can compute the maximum discharge capacity, as well as the Coulombic efficiency
        q_t = valid_segments[0][0]['cycled_charge'] / 3600.  # Convert to Amp-hours
        ce = valid_segments[0][0]['cycled_charge'] / valid_segments[1][0]['cycled_charge']

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
