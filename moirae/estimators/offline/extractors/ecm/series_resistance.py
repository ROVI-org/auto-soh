"""
Defines functionality for R0 extraction
"""
from typing import List, Optional, Union
from typing_extensions import Self
from warnings import warn

import numpy as np
import pandas as pd

from battdat.data import BatteryDataset
from battdat.postprocess.tagging import AddState
from battdat.postprocess.integral import StateOfCharge
from battdat.schemas.column import ChargingState, ControlMethod

from moirae.models.ecm.components import MaxTheoreticalCapacity
from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset
from moirae.estimators.offline.DataCheckers.RPT import FullHPPCDataChecker as HPPCChecker
from moirae.estimators.offline.extractors.base import BaseExtractor, ExtractedParameter


def compute_r0(raw_data: pd.DataFrame, valid_idx: List[int]) -> List[float]:
    """
    Computes series resistance values based on the differences of voltage and of current between the indices provided
    and the indices that precede them

    Args:
        raw_data: raw battery data, provided as a dataframe
        valid_idx: list of indices to be used for R0 extraction; assumes current differences to be large 
    """
    # Get the values at the present timestep
    end_voltage = raw_data.loc[valid_idx, 'voltage'].to_numpy()
    end_current = raw_data.loc[valid_idx, 'current'].to_numpy()

    # Get the values from the previous timestep
    beg_voltage = raw_data.loc[[val - 1 for val in valid_idx], 'voltage'].to_numpy()
    beg_current = raw_data.loc[[val - 1 for val in valid_idx], 'current'].to_numpy()

    # Compute relevant differences
    delta_V = end_voltage - beg_voltage
    delta_I = end_current - beg_current

    # Compute R0
    r0_vals = delta_V / delta_I

    if not np.all(r0_vals > 0.):
        raise ValueError('Negative values of series resistance have been found!')
    
    return r0_vals


class R0Extractor(BaseExtractor):
    """
    Estimates the values of the series resistance, as extracted from an HPPC test.

    Args:
        hppc_checker: data checker for the HPPC cycle
        dt_max: maximum allowed timestep (in seconds) between relevant successivedatapoints to be used for R0
            computation; defaults to 20 ms
        dcurr_min: minimum allowed current difference (in Amps) between relevant successive datapoints to be used for R0
            computation; defaults to 100 mA
    """
    def __init__(self,
                 hppc_checker: HPPCChecker,
                 dt_max: float = 0.02,
                 dcurr_min: float = 0.1) -> None:
        self.data_checker = hppc_checker
        self.dt_max = dt_max
        self.dcurr_min = dcurr_min

    @property
    def data_checker(self) -> HPPCChecker:
        return self._data_checker

    @data_checker.setter
    def data_checker(self, checker: HPPCChecker) -> None:
        if not isinstance(checker, HPPCChecker):
            raise TypeError('Data checker must be a FullHPPCDataChecker!')
        self._data_checker = checker

    @property
    def capacity(self) -> float:
        return self._data_checker.capacity

    @property
    def coul_eff(self) -> float:
        return self._data_checker.coulombic_efficiency

    @property
    def rest_current_threshold(self) -> float:
        return self._data_checker.rest_current_threshold

    @classmethod
    def init_from_basics(self,
                         capacity: Union[float, MaxTheoreticalCapacity],
                         dt_max: float = 0.02,
                         dcurr_min: float = 0.1,
                         coulombic_efficiency: float = 1.0,
                         min_delta_soc: float = 0.1,
                         min_pulses: int = 1,
                         ensure_bidirectional: bool = False,
                         min_number_of_rests: int = 1,
                         min_rest_duration: float = 600.,
                         rest_current_threshold: float = 1.0e-04) -> Self:
        """
        Helper function to initialize extractor from smallest elements.

        Args:
            capaciy: cell capacity in Amp-hours
            dt_max: maximum allowed timestep (in seconds) between relevant successivedatapoints to be used for R0
                computation; defaults to 20 ms
            dcurr_min: minimum allowed current difference (in Amps) between relevant successive datapoints to be used
                for R0 computation; defaults to 100 mA
            coulombic_efficiency: coulombic efficiency; defaults to 100%
            min_delta_soc: minimum range off SOC values that must be attained during the cycle; defaults to 10%
            min_pulses: mimimum number of pulses that must be observed; defaults to 1
            ensure_bidirectional: flag to ensure pulses are bidirectional; defaults to False
            min_number_of_rests: minimum number of rests that must be present; defaults to 1
            min_rest_duration: minimum duration of a given rest in seconds; defaults to 10 min
            rest_current_threshold: current threshold for a segment to be considered a rest; defaults to 0.1 mA
        """
        # Initialize HPPC checker
        checker = HPPCChecker(capacity=capacity,
                              coulombic_efficiency=coulombic_efficiency,
                              min_delta_soc=min_delta_soc,
                              min_pulses=min_pulses,
                              ensure_bidirectional=ensure_bidirectional,
                              min_number_of_rests=min_number_of_rests,
                              min_rest_duration=min_rest_duration,
                              rest_current_threshold=rest_current_threshold)
        
        return R0Extractor(hppc_checker=checker, dt_max=dt_max, dcurr_min=dcurr_min)

    def identify_valid_current_changes(self, data: Union[pd.DataFrame, BatteryDataset]) -> List[int]:
        """
        Function to identify points in data where changes in current larger than the specified threshold occur, so that
        they can be used for computation of R0

        Args:
            data: data containing significant changes in current over short periods of time

        Returns:
            indices where these points happen (if index `i` is returned, the change in current happened from indes `i-1`
            to `i`)
        """
        # Ensure dataset
        data = ensure_battery_dataset(data=data)
        raw_data = data.tables.get('raw_data')

        # Compute relevant differences
        delta_curr = raw_data['current'].diff().abs() >= self.dcurr_min
        delta_t = raw_data['test_time'].diff() <= self.dt_max

        # Get valid indices
        valid_idx = np.logical_and(delta_curr, delta_t)
        valid_idx = valid_idx[valid_idx == True]
        valid_idx = valid_idx.index.to_list()

        if len(valid_idx) == 0:
            warn(f'No valid current changes of magnitude >= {self.dcurr_min:.2f} Amps found in a period shorter than '
                 f'{1000 * self.dt_max:.1f} milliseconds!')

        return valid_idx

    def validate_indices(self, data: Union[pd.DataFrame, BatteryDataset], indices: List[int]) -> List[int]:
        """
        Auxiliary function that refines a list of indices to be used for R0 extraction.

        It compares the current and time of the indices provided with those that precede them, and assembles a final
        list of indices containing a subset of valid ones

        Args:
            data: data to be used
            indices: list of indices to check

        Returns:
            refined list of indices
        """
        # Ensure dataset
        data = ensure_battery_dataset(data=data)
        raw_data = data.tables.get('raw_data')

        # Construt return list
        valid_idx = []

        for index in indices:
            if index == raw_data.index.to_list()[0]:  # If it's the first entry in the dataframe, skip it
                continue
            delta_t = raw_data.loc[index, 'test_time'] - raw_data.loc[index - 1, 'test_time']
            delta_I = raw_data.loc[index, 'current'] - raw_data.loc[index - 1, 'current']
            if (delta_t <= self.dt_max) and (delta_I >= self.dcurr_min):
                valid_idx += [index]

        if len(valid_idx) == 0:
            warn(f'No valid current changes of magnitude >= {self.dcurr_min:.2f} Amps found in a period shorter than '
                 f'{1000 * self.dt_max:.1f} milliseconds!')

        return valid_idx

    def compute_parameters(self,
                           data: Union[pd.DataFrame, BatteryDataset],
                           indices: Optional[List[int]] = None,
                           start_soc: float = 0.0) -> ExtractedParameter:
        """
        Function to compute the parameters to be extracted. 

        If a list of indices is provided, it will validate them and use them for computation of R0 values. Otherwise,
        the list will be generated internally.

        Args:
            data: data to be used to compute R0
            indices: list of indices from which R0 will be extracted; if not provided, will be automatically generated
            start_soc: SOC at the beginning of the reported data; defaults to 0.

        Returns:
            extracted R0
        """
        # Ensure dataset
        data = ensure_battery_dataset(data=data)
        raw_data = data.tables.get('raw_data')

        # Check the indices
        if indices is None:
            validated_idx = self.identify_valid_current_changes(data=data)
        else:
            validated_idx = self.validate_indices(data=data, indices=indices)

        # Enhance the raw data if necessary, so we can compute SOC
        if 'state' not in raw_data.columns:
            AddState(rest_curr_threshold=self.rest_current_threshold).enhance(data=raw_data)
        if 'CE_adjusted_charge' not in raw_data.columns:
            StateOfCharge(coulombic_efficiency=self.coul_eff).enhance(data=raw_data)

        # Compute SOC
        soc_end = raw_data.loc[validated_idx, 'CE_adjusted_charge'].to_numpy() / self.capacity
        soc_beg = raw_data.loc[[idx - 1 for idx in validated_idx], 'CE_adjusted_charge'].to_numpy() / self.capacity
        soc_avg = start_soc + (soc_end + soc_beg) / 2.

        # Compute R0 values
        r0_vals = compute_r0(raw_data=raw_data, valid_idx=validated_idx)

        # Assemble the extracted parameter
        r0_param = ExtractedParameter(value=r0_vals,
                                      soc_level=soc_avg,
                                      units='Ohm')

        return r0_param

    def extract(self, data: Union[pd.DataFrame, BatteryDataset], start_soc: float = 0.) -> ExtractedParameter:
        """
        Extracts R0

        Args:
            data: data to be used to compute R0
            start_soc: SOC at the beginning of the reported data; defaults to 0.

        Returns:
            extracted R0
        """
        # Check data
        data = self._data_checker.check(data=data)
        raw_data = data.tables.get('raw_data')

        # Since the data has been checked, we can find the location of pulses and rests
        pulses = raw_data[raw_data['method'] == ControlMethod.pulse]
        rests = raw_data[raw_data['state'] == ChargingState.rest]

        # Now, let us use the beginning of each of these segments to get our valid indices
        indices = pulses.drop_duplicates('substep_index', keep='first').index.to_list()
        indices += rests.drop_duplicates('step_index', keep='first').index.to_list()

        return self.compute_parameters(data=data, indices=indices, start_soc=start_soc)
