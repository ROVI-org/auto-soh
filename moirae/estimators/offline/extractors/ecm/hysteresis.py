"""
Defines the hysteresis extractor
"""
from typing import List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd

from battdat.data import BatteryDataset
from battdat.schemas.column import ChargingState
from battdat.postprocess.tagging import AddState, AddSteps
from battdat.postprocess.integral import StateOfCharge

from moirae.models.ecm.components import (MaxTheoreticalCapacity,
                                          Resistance,
                                          OpenCircuitVoltage,
                                          RCComponent,
                                          HysteresisParameters)
from moirae.models.ecm.advancedSOH import ECMASOH
from moirae.models.ecm.transient import ECMTransientVector
from moirae.models.ecm import EquivalentCircuitModel as ECM
from moirae.interface import run_model

from moirae.estimators.offline.extractors.base import BaseExtractor, ExtractedParameter
from moirae.estimators.offline.DataCheckers.RPT import CapacityDataChecker
from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset


class ExtractedHysteresis(ExtractedParameter):
    """
    Definition for extracted hysteresis, which also accounts for the time step at which the value was extracted
    Args:
        value: extracted values
        units: unit of measurement for values
        soc_level: SOC level for values, if appropriate
        step_time: time (in seconds) since the beginning of the charge, discharge, or rest step from which hysteresis
            is being extracted
        adjusted_curr: adjusted current value, that is, current * CE / capacity, measures in Hz
    """
    step_time: Union[List, np.ndarray]
    adjusted_curr: Union[List, np.ndarray]


class HysteresisExtractor(BaseExtractor):
    """
    Extracts the open circuit voltage (OCV) from a 100% depth-of-discharge (DoD), low C-rate cycle

    Args:
        capacity: capacity of the cell in Amp-hours
        ocv: cell open circuit voltage, in Volts
        series_resistance: resistance to be used when computing IR terms; defaults to zero resistance
        coulombic_efficiency: coulombic efficiency of the cell; defaults to 100%
        rc_elements: list of RC components; defaults to no RC pairs
        data_checker: data checker for capacity check tests
    """
    def __init__(self,
                 capacity: Union[float, MaxTheoreticalCapacity],
                 ocv: OpenCircuitVoltage,
                 series_resistance: Resistance = Resistance(base_values=0),
                 rc_elements: List[RCComponent] = [],
                 coulombic_efficiency: float = 1.,
                 data_checker: CapacityDataChecker = CapacityDataChecker()):
        self.capacity = capacity
        self.ocv = ocv
        self.series_resistance = series_resistance
        self.rc_elements = rc_elements
        self.coulombic_efficiency = coulombic_efficiency
        self.data_checker = data_checker

    @classmethod
    def init_from_basics(self,
                         capacity: Union[float, MaxTheoreticalCapacity],
                         ocv: OpenCircuitVoltage,
                         coulombic_efficiency: float = 1.,
                         series_resistance: Union[Resistance, float] = 0.,
                         rc_elements: List[RCComponent] = [],
                         voltage_limits: Optional[Tuple[float, float]] = None,
                         max_C_rate: float = 0.1,
                         voltage_tolerance: float = 0.001) -> Self:
        """
        Helper function to initialize OCV extractor for basic information

        Args:
            capacity: cell capacity, in Amp-hours
            ocv: cell open circuit voltage, in Volts
            coulombic_efficiency: coulombic efficiency of the cell; defaults to 100%
            series_resistance: resistance to be used when computing IR terms; defaults to zero resistance
            rc_elements: list of RC components; defaults to no RC pairs
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
        return HysteresisExtractor(capacity=capacity,
                                   ocv=ocv,
                                   coulombic_efficiency=coulombic_efficiency,
                                   series_resistance=series_resistance,
                                   rc_elements=rc_elements,
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
            self._r0 = resistance.model_copy(deep=True)
        elif isinstance(resistance, float):
            self._r0 = Resistance(base_values=[resistance, resistance])
        else:
            raise ValueError('Resistance must be provided as a float or as a Resistance object!')

    @property
    def rc_elements(self) -> List[RCComponent]:
        return self._rc_pairs

    @rc_elements.setter
    def rc_elements(self, rc_pairs: List[RCComponent]):
        self._rc_pairs = [rc.model_copy(deep=True) for rc in rc_pairs]

    @property
    def ocv(self) -> OpenCircuitVoltage:
        return self._ocv

    @ocv.setter
    def ocv(self, func: OpenCircuitVoltage):
        if not isinstance(func, OpenCircuitVoltage):
            raise ValueError('OCV must be of proper type!')
        self._ocv = func.model_copy(deep=True)

    @property
    def hypothetical_asoh(self) -> ECMASOH:
        """
        Auxiliary property that builds a corresponding aSOH but with no hysteresis
        """
        asoh = ECMASOH(q_t=MaxTheoreticalCapacity(base_values=self._capacity),
                       ce=self._ce,
                       ocv=self._ocv,
                       r0=self._r0,
                       rc_elements=self._rc_pairs,
                       h0=HysteresisParameters(base_values=0.))
        return asoh

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
        Auxiliary function to help identify useful steps to be used when computing hysteresis. Steps returned are either
        charge or discharge steps with long duration, so that, by the end of it, the RC pairs will be saturated.

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
            include_step = False  # Boolean flag to indicate whether step should be included
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
                           start_soc: float = 0.,
                           valid_steps: Optional[List[int]] = None) -> ExtractedHysteresis:
        """
        Computes Hysteresis assuming it is the difference between observed terminal voltage and expected (OCV + IR-drop)
        terminal voltage

        Args:
            data: data to be used, assumed to have already passed necessary checks
            start_soc: SOC at the beginning of the provided data
            valid_steps: steps that will be used to extract instantaneous hysteresis

        Returns:
            extracted hysteresis
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

        # Now, for each identified step, we will simulate the cell with an aSOH that assumes no hysteresis, and assume
        # the difference between the simulated and measured voltage corresponds to the instantaneous hysteresis.
        # Note that, for that, we also make the assumption that, at the beginning of each segment, the RC components
        # have no overpotential
        hyst_vals = []
        soc_vals = []
        step_time = []
        adj_curr = []

        # Iterate through steps
        for step_id in valid_steps:
            # Get data
            step_raw = raw_data[raw_data['step_index'] == step_id]
            step_data = BatteryDataset.make_cell_dataset(raw_data=step_raw)
            # Prepare to simulate
            soc = start_soc + (step_raw['CE_adjusted_charge'].to_numpy() / self._capacity)
            transient0 = ECMTransientVector.from_asoh(asoh=self.hypothetical_asoh)
            transient0.soc = np.atleast_2d(soc[0])
            # Run ECM model
            simulated_cell = run_model(model=ECM(),
                                       dataset=step_data,
                                       asoh=self.hypothetical_asoh,
                                       state_0=transient0)
            # Instantaneous hysteresis computed as the difference
            inst_hyst = step_raw['voltage'].to_numpy() - simulated_cell['terminal_voltage'].to_numpy()
            times = step_raw['test_time'].to_numpy().flatten()
            times = times - times[0]
            currs = step_raw['current'].to_numpy() * self._ce / (self._capacity * 3600.)
            # Add to running values
            hyst_vals += inst_hyst.tolist()
            soc_vals += soc.tolist()
            step_time += times.tolist()
            adj_curr += currs.flatten().tolist()

        return ExtractedHysteresis(value=np.array(hyst_vals),
                                   soc_level=np.array(soc_vals),
                                   units='Volt',
                                   step_time=np.array(step_time),
                                   adjusted_curr=np.array(adj_curr)
                                   )

    def extract(self,
                data: Union[pd.DataFrame, BatteryDataset],
                start_soc: float = 0.) -> ExtractedHysteresis:
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
