"""
Defines necessary scripts for RC extraction
"""
from typing import List, Tuple, Union
from typing_extensions import Self
from warnings import warn

import numpy as np
import pandas as pd

from battdat.data import BatteryDataset
from battdat.postprocess.tagging import AddState, AddSteps
from battdat.postprocess.integral import StateOfCharge
from battdat.schemas.column import ChargingState

from moirae.models.ecm.components import MaxTheoreticalCapacity
from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset
from moirae.estimators.offline.DataCheckers.RPT import RestDataChecker
from moirae.estimators.offline.extractors.base import BaseExtractor, ExtractedParameter

from .utils import compute_I_RCs, fit_exponential_decays


def compute_RC_from_exp_params(amplitudes: np.ndarray,
                               taus: np.ndarray,
                               prev_time: np.ndarray,
                               prev_current: np.ndarray,
                               prev_q0: Union[float, np.ndarray] = 0.) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auxiliary function to convert results from exponential decay fit into resistance and capacitance terms based on the
    step that precedes the rest used for fitting

    Args:
        amplitudes: amplitudes of the exponential decays
        taus: relaxation times of the exponential decays
        prev_time: timestamps of the previous step
        prev_curr: current of the previous step
        prev_q0s: charge in the capacitors at the beginning of the previous step; defaults to 0.0

    Returns:
        tupple of arrays containing the RC values, where the first array corresponds to the resistance(s), and the
        second, to the capacitance(s)
    """
    # Compute the current through the resistor at the end of the previous step
    i_rcs = compute_I_RCs(total_current=prev_current,
                          timestamps=prev_time,
                          tau_values=taus,
                          qc0s=prev_q0)
    # Now, the resistances are easy
    r_vals = amplitudes / i_rcs
    # Finally, the capacitances
    c_vals = taus / r_vals

    return r_vals, c_vals


class RCExtractor(BaseExtractor):
    """
    Estimate the values of resistor-capacitor couples (RC) for an ECM as a function of state of charge (SOC).

    Requires data containing extended rest periods.

    Args:
        hppc_checker: Data checker for HPPC
        min_prev_duration: Minimum duration (in seconds) of the step that precedes the rest(s). Defaults to 5 minutes
    """
    def __init__(self,
                 rest_checker: RestDataChecker):
        self.data_checker = rest_checker

    @property
    def data_checker(self) -> RestDataChecker:
        return self._data_checker

    @data_checker.setter
    def data_checker(self, checker: RestDataChecker) -> None:
        if not isinstance(checker, RestDataChecker):
            raise TypeError('Data checker must be a FullHPPCDataChecker!')
        self._data_checker = checker

    @property
    def capacity(self) -> float:
        return self._data_checker.capacity

    @capacity.setter
    def capacity(self, value: Union[float, MaxTheoreticalCapacity]):
        self._data_checker.capacity = value

    @property
    def coul_eff(self) -> float:
        return self._data_checker.coulombic_efficiency

    @coul_eff.setter
    def coul_eff(self, value: float):
        self._data_checker.coulombic_efficiency = value

    @property
    def min_prev_duration(self) -> float:
        return self._data_checker.min_prev_duration

    @property
    def rest_current_threshold(self) -> float:
        return self._data_checker.rest_current_threshold

    @classmethod
    def init_from_basics(self,
                         capacity: Union[float, MaxTheoreticalCapacity],
                         min_prev_duration: float = 300.,
                         coulombic_efficiency: float = 1.0,
                         min_delta_soc: float = 0.1,
                         min_number_of_rests: int = 1,
                         min_rest_duration: float = 600.,
                         rest_current_threshold: float = 1.0e-04) -> Self:
        """
        Helper function to initialize extractor from smallest elements.

        Args:
            capaciy: cell capacity in Amp-hours
            min_prev_duration: minimum duration (in seconds) of the step that precedes the rest(s); defaults to 5 min
            coulombic_efficiency: coulombic efficiency; defaults to 100%
            min_delta_soc: minimum range off SOC values that must be attained during the cycle; defaults to 10%
            min_number_of_rests: minimum number of rests that must be present; defaults to 1
            min_rest_duration: minimum duration of a given rest in seconds; defaults to 10 min
            rest_current_threshold: current threshold for a segment to be considered a rest; defaults to 0.1 mA
        """
        # Initialize HPPC checker
        checker = RestDataChecker(capacity=capacity,
                                  coulombic_efficiency=coulombic_efficiency,
                                  min_delta_soc=min_delta_soc,
                                  min_number_of_rests=min_number_of_rests,
                                  min_rest_duration=min_rest_duration,
                                  min_prev_duration=min_prev_duration,
                                  rest_current_threshold=rest_current_threshold)
        
        return RCExtractor(rest_checker=checker)

    def identify_rests_steps(self, data: Union[pd.DataFrame, BatteryDataset]) -> List[int]:
        """
        Function to identify valid rest periods, as well as the previous step.

        Args:
            data: data assumed to contain enough valid rest periods

        Returns:
            a list of the step indices of valid rests found
        """
        # Ensure compliance with the BatteryDataToolkit
        data = ensure_battery_dataset(data=data)
        raw_data = data.tables.get('raw_data')

        # Make sure it has what we need
        if 'state' not in raw_data.columns:
            AddState(rest_curr_threshold=self._data_checker.rest_current_threshold).enhance(data=raw_data)
        if 'step_index' not in raw_data.columns:
            AddSteps().enhance(data=raw_data)

        # Aggregate rest data
        rest_data = raw_data[raw_data['state'] == ChargingState.rest]

        # Initialize return list
        rest_step_idx = []

        # Go though rests
        for rest_step, rest in rest_data.groupby('step_index'):
            rest_dur = rest['test_time'].iloc[-1] - rest['test_time'].iloc[0]
            if rest_dur < self._data_checker.min_rest_duration:
                continue
            # Now, get the previous step
            prev_step = raw_data[raw_data['step_index'] == rest_step - 1]
            if len(prev_step) == 0:  # If the first rest is at the very beginning of the cycle
                continue
            prev_dur = prev_step['test_time'].iloc[-1] - prev_step['test_time'].iloc[0]
            if prev_dur >= self.min_prev_duration:
                rest_step_idx.append(rest_step)

        # Check if we found what we needed
        if len(rest_step_idx) == 0:
            warn('No suitable rests found!')

        return rest_step_idx

    def compute_parameters(self,
                           data: Union[pd.DataFrame, BatteryDataset],
                           n_rc: int = 1,
                           start_soc: float = 0.0) -> List[Tuple[ExtractedParameter]]:
        """
        Computes the RC parameters from the data.

        Args:
            data: data to be used when fitting
            n_rc: number of RC components to fit
            start_soc: SOC at the beginning of the reported data; defaults to 0.

        Returns:
            list of RC-tuples of extracted parameters, one entry for each desired RC component
        """
        # Ensure formatting
        data = ensure_battery_dataset(data=data)
        raw_data = data.tables.get('raw_data')
        # Make sure we can easily compute the SOC
        if 'CE_adjusted_charge' not in raw_data.columns:
            StateOfCharge(coulombic_efficiency=self.coul_eff).enhance(raw_data)

        # Initialize array to collect values
        r_vals = []
        c_vals = []
        socs = []

        # Get valid rest steps
        rest_step_idx = self.identify_rests_steps(data=data)

        for step in rest_step_idx:
            rest_data = raw_data[raw_data['step_index'] == step]
            # Get voltage, time, and soc
            soc = start_soc + (rest_data['CE_adjusted_charge'].iloc[0] / self.capacity)
            timestamps = rest_data['test_time'].to_numpy()
            voltage = rest_data['voltage'].to_numpy()
            # Get exponential decay parameters
            amplitudes, taus = fit_exponential_decays(time=timestamps,
                                                      measurements=voltage,
                                                      n_exp=n_rc)
            # Get the previous step timestamps and current
            prev_step = raw_data[raw_data['step_index'] == step - 1]
            prev_time = prev_step['test_time'].to_numpy()
            prev_curr = prev_step['current'].to_numpy()
            # Compute R and C from these
            r, c = compute_RC_from_exp_params(amplitudes=amplitudes,
                                              taus=taus,
                                              prev_time=prev_time,
                                              prev_current=prev_curr)
            # Append to existing list
            r_vals.append(r)
            c_vals.append(c)
            socs.append(soc)

        # Convert to numpy arrays for ease of processing
        r_vals = np.atleast_2d(np.array(r_vals)).T
        c_vals = np.atleast_2d(np.array(c_vals)).T
        socs = np.array(socs)

        # Assemble return list
        rc_params = []

        for i in range(n_rc):
            resistance = ExtractedParameter(value=r_vals[i, :].flatten(),
                                            soc_level=socs.copy(),
                                            units='Ohm')
            capacitance = ExtractedParameter(value=c_vals[i, :].flatten(),
                                             soc_level=socs.copy(),
                                             units='Farad')
            rc_params.append((resistance, capacitance))

        return rc_params

    def extract(self, data: Union[pd.DataFrame, BatteryDataset],
                n_rc: int = 1,
                start_soc: float = 0.0) -> List[Tuple[ExtractedParameter]]:
        """
        Computes the RC parameters from the data after it passes necessary checks

        Args:
            data: data to be used when fitting
            n_rc: number of RC components to fit
            start_soc: SOC at the beginning of the reported data; defaults to 0.

        Returns:
            list of RC-tuples of extracted parameters, one entry for each desired RC component
        """
        data = self._data_checker.check(data=data)
        return self.compute_parameters(data=data, n_rc=n_rc, start_soc=start_soc)
