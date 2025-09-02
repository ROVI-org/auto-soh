"""
Defines the hysteresis extractor
"""
from typing import List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd

from battdat.data import BatteryDataset
from battdat.postprocess.tagging import AddState, AddSteps
from battdat.postprocess.integral import StateOfCharge

from moirae.models.ecm.components import MaxTheoreticalCapacity, Resistance, OpenCircuitVoltage
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
        exponential_factor: values of dynamic hysteresis model exponential factor, that is, 1 minus the exponent of the
            multiplication of current by coulombic efficiency by delta time since the beginninng of the step divided by
            capacity
        step_time: time (in seconds) since the beginning of the charge, discharge, or rest step from which hysteresis
            is being extracted
    """
    exponential_factor: Union[List, np.ndarray]


class HysteresisExtractor(BaseExtractor):
    """
    Extracts the open circuit voltage (OCV) from a 100% depth-of-discharge (DoD), low C-rate cycle

    Args:
        data_checker: data checker for capacity check tests
        capacity: capacity of the cell in Amp-hours
        ocv: cell open circuit voltage, in Volts
        gamma: hysteresis gamma parameter, proportionality constant related to how quickly instantaneous hysteresis
        coulombic_efficiency: coulombic efficiency of the cell; defaults to 100%
        series_resistance: resistance to be used when computing IR terms; defaults to zero resistance
    """
    def __init__(self,
                 capacity: Union[float, MaxTheoreticalCapacity],
                 ocv: OpenCircuitVoltage,
                 gamma: float = 50.,
                 coulombic_efficiency: float = 1.,
                 series_resistance: Union[Resistance, float] = 0.,
                 data_checker: CapacityDataChecker = CapacityDataChecker()):
        self.data_checker = data_checker
        self.capacity = capacity
        self.gamma = gamma
        self.ocv = ocv
        self.coulombic_efficiency = coulombic_efficiency
        self.series_resistance = series_resistance

    @classmethod
    def init_from_basics(self,
                         capacity: Union[float, MaxTheoreticalCapacity],
                         ocv: OpenCircuitVoltage, 
                         coulombic_efficiency: float = 1.,
                         gamma: float = 50.,
                         series_resistance: Union[Resistance, float] = 0.,
                         voltage_limits: Optional[Tuple[float, float]] = None,
                         max_C_rate: float = 0.1,
                         voltage_tolerance: float = 0.001) -> Self:
        """
        Helper function to initialize OCV extractor for basic information

        Args:
            capacity: cell capacity, in Amp-hours
            ocv: cell open circuit voltage, in Volts
            coulombic_efficiency: coulombic efficiency of the cell; defaults to 100%
            gamma: hysteresis gamma parameter, proportionality constant related to how quickly instantaneous hysteresis
            approaches hysteresis limit
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
        return HysteresisExtractor(capacity=capacity,
                                   ocv=ocv,
                                   gamma=gamma,
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
    def gamma(self) -> float:
        """Hysteresis gamma paramter"""
        return self._gamma

    @gamma.setter
    def gamma(self, value: float):
        if value <= 0.:
            raise ValueError("Hysteresis Gamma must be positive")
        self._gamma = value

    @property
    def kappa(self) -> float:
        """Hysteresis Kappa parameters, that is, factor that determines rate of exponential approach"""
        return self._gamma * self._ce / (3600 * self._capacity)

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

    def compute_parameters(self,
                           data: Union[pd.DataFrame, BatteryDataset],
                           start_soc: float = 0.) -> ExtractedHysteresis:
        """
        Computes Hysteresis assuming it is the difference between observed terminal voltage and expected (OCV + IR-drop)
        terminal voltage

        Args:
            data: data to be used, assumed to have already passed necessary checks
            start_soc: SOC at the beginning of the provided data

        Returns:
            extracted hysteresis
        """
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

        # Compute SOC at every timestep, as well as corresponding OCV and IR-drop
        soc = start_soc + (raw_data['CE_adjusted_charge'].to_numpy() / self.capacity)
        ocv = self.ocv(soc=soc).flatten()
        ir_drop = (self._r0.get_value(soc=soc).flatten() * raw_data['current'].to_numpy())

        # From this, we can compute the "expected" terminal voltage
        expected_vt = ocv + ir_drop

        # Our assumption is that the hysteresis is equivalent to the difference between observed and expected terminal
        # voltage, which is an okay assumption, since we are looking at a capacity check cycle
        hyst = np.abs(raw_data['voltage'].to_numpy() - expected_vt)

        # Now, we need to compute the time since the beginning of each step
        step_times = []
        for step_id, step_data in raw_data.groupby('step_index'):
            dt = step_data['test_time'].to_numpy() - step_data['test_time'].iloc[0]
            step_times += dt.tolist()
        # Convert that to numpy array
        step_time=np.array(step_times)
        # Now, multiply by the relevant current and by kappa to get the exponential prefactor
        exp_fact = 1. - np.exp(-abs(self.kappa * raw_data['current'].to_numpy() * step_time)).flatten()

        # Prepare return object
        extracted_hyst = ExtractedHysteresis(value=hyst,
                                             soc_level=soc,
                                             exponential_factor=exp_fact,
                                             units='Volt')
        
        return extracted_hyst

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
