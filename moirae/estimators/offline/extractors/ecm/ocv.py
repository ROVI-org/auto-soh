"""
Defines the OCV extractor
"""
from typing import List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd

from battdat.data import BatteryDataset
from battdat.schemas.column import ChargingState
from battdat.postprocess.tagging import AddState, AddSteps
from battdat.postprocess.integral import StateOfCharge

from moirae.models.ecm.components import MaxTheoreticalCapacity, Resistance
from moirae.estimators.offline.extractors.base import BaseExtractor, ExtractedParameter
from moirae.estimators.offline.DataCheckers.RPT import CapacityDataChecker
from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset


class OCVExtractor(BaseExtractor):
    """
    Extracts the open circuit voltage (OCV) from a 100% depth-of-discharge (DoD), low C-rate cycle

    Args:
        data_checker: data checker for capacity check tests
        capacity: capacity of the cell in Amp-hours
        coulombic_efficiency: coulombic efficiency of the cell; defaults to 100%
        series_resistance: resistance to be used when computing IR terms; defaults to zero resistance
    """
    def __init__(self,
                 capacity: Union[float, MaxTheoreticalCapacity],
                 coulombic_efficiency: float = 1.,
                 series_resistance: Union[Resistance, float] = 0.,
                 data_checker: CapacityDataChecker = CapacityDataChecker()):
        self.data_checker = data_checker
        self.capacity = capacity
        self.coulombic_efficiency = coulombic_efficiency
        self.series_resistance = series_resistance

    @classmethod
    def init_from_basics(self,
                         capacity: Union[float, MaxTheoreticalCapacity],
                         coulombic_efficiency: float = 1.,
                         series_resistance: Union[Resistance, float] = 0.,
                         voltage_limits: Optional[Tuple[float, float]] = None,
                         max_C_rate: float = 0.1,
                         voltage_tolerance: float = 0.001) -> Self:
        """
        Helper function to initialize OCV extractor for basic information

        Args:
            capacity: cell capacity, in Amp-hours
            coulombic_efficiency: coulombic efficiency of the cell; defaults to 100%
            series_resistance: resistance to be used when computing IR terms; defaults to zero resistance
            voltage_limits: Tuple of (min_voltage, max_voltage) to check against; if not provided, does not check
                voltage
            max_C_rate: Maximum approximate C-rate for the cycle to be considered a capacity check; deafults to C/10
            voltage_tolerance: Tolerance for voltage limits, defaults to 1 mV

        Returns:
            instance of OCVExtractor
        """
        checker = CapacityDataChecker(voltage_limits=voltage_limits,
                                      max_C_rate=max_C_rate,
                                      voltage_tolerance=voltage_tolerance)
        return OCVExtractor(capacity=capacity,
                            coulombic_efficiency=coulombic_efficiency,
                            series_resistance=series_resistance,
                            data_checker=checker)

    @property
    def data_checker(self) -> CapacityDataChecker:
        return self._data_checker

    @data_checker.setter
    def data_checker(self, checker: CapacityDataChecker):
        if not isinstance(checker, CapacityDataChecker):
            raise TypeError('Data checker must be a CapacityDataChecker object!')
        self._data_checker = checker

    @property
    def capacity(self) -> float:
        """
        Returns capacity in Amp-hours
        """
        return self._capacity

    @capacity.setter
    def capacity(self, value: Union[float, MaxTheoreticalCapacity]):
        if isinstance(value, MaxTheoreticalCapacity):
            self._capacity = value.amp_hour.item()
        elif isinstance(value, (int, float)):
            if value <= 0:
                raise ValueError("Capacity must be a positive number!")
            self._capacity = float(value)
        else:
            raise TypeError("Capacity must be a float or MaxTheoreticalCapacity object!")

    @property
    def coulombic_efficiency(self) -> float:
        """Coulombic efficiency of the cell"""
        return self._ce

    @coulombic_efficiency.setter
    def coulombic_efficiency(self, value: float):
        if value < 0 or value > 1:
            raise ValueError("Coulombic efficiency must be between 0 and 1.")
        self._ce = value

    @property
    def series_resistance(self) -> Resistance:
        return self._r0

    @series_resistance.setter
    def series_resistance(self, resistance: Union[Resistance, float]):
        """
        Sets series resistance
        """
        if isinstance(resistance, Resistance):
            self._r0 = resistance
        elif isinstance(resistance, float):
            self._r0 = Resistance(base_values=[resistance, resistance])
        else:
            raise ValueError('Resistance must be provided as a float or as a Resistance object!')

    @property
    def voltage_limits(self) -> Union[Tuple[float, float], None]:
        """
        Voltage limits of the cell, in volts
        """
        return self._data_checker.voltage_limits

    @property
    def max_current(self) -> float:
        """
        Maximum allowed current (in Amps) based on specified C-rate
        """
        return (self.capacity * self._data_checker.max_C_rate)

    @property
    def min_segment_duration(self) -> float:
        """
        Minimum duration (in seconds) for a charge or discharge segment to be considered valid
        """
        return 3600. / self._data_checker.max_C_rate

    @property
    def volt_tol(self) -> float:
        """
        Absolute voltage tolerance for voltage limits, in volts
        """
        return self._data_checker.voltage_tolerance

    def identify_valid_steps(self, data: Union[pd.DataFrame, BatteryDataset]) -> List[int]:
        """
        Auxiliary function to help identify useful steps to be used when computing OCV. Steps returned are either charge
        or discharge steps with low current and long duration.

        Args:
            data: data to be used when extracting OCV; assumed to have already passed necessary checks
        """
        # Ensure battery dataset
        data = ensure_battery_dataset(data=data)
        raw_data = data.tables.get('raw_data')

        # Postprocessing what is needed
        if 'state' not in raw_data.columns:
            AddState().enhance(raw_data)
        if 'step_index' not in raw_data.columns:
            AddSteps().enhance(raw_data)

        # Initialize list to keep valid steps
        step_idx = []

        # We want to look at the steps that are either charging or discharging, as it is in them that the SOC changes
        # We also need to make sure each of these steps is adequate for what we want
        segments = raw_data[raw_data['state'].isin([ChargingState.charging, ChargingState.discharging])]
        for step_id, step in segments.groupby('step_index'):
            include_step = False  # Boolean flag to 
            duration = step['test_time'].iloc[-1] - step['test_time'].iloc[0]
            if duration >= self.min_segment_duration:  # Segment lasts for a long time
                # Check if current is small enough throughout
                if np.all(step['current'].abs() <= self.max_current):
                    # Check for voltage limits
                    if self.voltage_limits is not None:
                        # If we must check the voltage limits
                        min_volt, max_volt = sorted(self.voltage_limits)
                        min_obs, max_obs = step['voltage'].min(), step['voltage'].max()
                        if np.allclose([min_volt, max_volt], [min_obs, max_obs], atol=self.volt_tol):
                            # If the voltage limits are reached
                            include_step = True
                    else:
                        include_step = True
            if include_step:
                step_idx.append(step_id)

        return step_idx

    def compute_parameters(self,
                           data: Union[pd.DataFrame, BatteryDataset],
                           valid_steps: Optional[List[int]] = None,
                           start_soc: float = 0.) -> ExtractedParameter:
        """
        Computes OCV from valid steps

        Args:
            data: data to be used, assumed to have already passed necessary checks
            valid_steps: list of steps to use; if not provided, will be computed
            start_soc: SOC at the beginning of the provided data

        Returns:
            extracted OCV
        """
        # Check if we need to compute valid steps
        if valid_steps is None:
            valid_steps = self.identify_valid_steps(data=data)

        # Ensure battery dataset
        data = ensure_battery_dataset(data=data)
        raw_data = data.tables.get('raw_data')

        # Compute SOC
        if 'state' not in raw_data.columns:
            AddState().enhance(raw_data)
        if 'step_index' not in raw_data.columns:
            AddSteps().enhance(raw_data)
        if 'CE_adjusted_charge' not in raw_data.columns:
            StateOfCharge(coulombic_efficiency=self._ce).enhance(data=raw_data)

        # Initialize OCV and SOC lists
        ocv_values = []
        soc_levels = []

        # Iterate through steps
        for step_id in valid_steps:
            step_data = raw_data[raw_data['step_index'] == step_id]
            # Get voltage and current
            voltage = step_data['voltage']
            current = step_data['current']
            # Get SOC
            soc = start_soc + (step_data['CE_adjusted_charge'].to_numpy() / self.capacity)
            # Get series resistance values
            r0_vals = self._r0.get_value(soc=soc).flatten()
            # Remove IR contribution
            ocv = voltage - (current * r0_vals)
            # Add to return values
            ocv_values += ocv.tolist()
            soc_levels += soc.tolist()
        
        # Assemble return dictionary
        extracted_ocv = ExtractedParameter(value=np.array(ocv_values),
                                           soc_level=np.array(soc_levels),
                                           units='Volt')
        return extracted_ocv

    def extract(self,
                data: Union[pd.DataFrame, BatteryDataset],
                start_soc: float = 0.) -> ExtractedParameter:
        """
        Extracts OCV

        Args:
            data: data to be used
            start_soc: SOC at the beginning of reported data

        Returns:
            extracted values of OCV
        """
        # Check data
        data = self._data_checker.check(data=data)

        return self.compute_parameters(data=data, start_soc=start_soc)
