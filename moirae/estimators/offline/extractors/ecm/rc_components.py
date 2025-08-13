"""
Defines necessary scripts for RC extraction
"""
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from battdat.data import BatteryDataset
from battdat.postprocess.integral import CapacityPerCycle, StateOfCharge
from battdat.postprocess.tagging import AddSteps, AddState

from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset
from moirae.estimators.offline.extractors.base import BaseExtractor
from moirae.models.ecm.components import MaxTheoreticalCapacity
from moirae.models.ecm.components import Resistance, Capacitance, RCComponent

from .utils import compute_I_RCs


class RCExtractor(BaseExtractor):
    """Estimate the values of a parallel resistor-capacitor couples (RC)
    for an ECM of a battery as a function of state of charge (SOC)

    Required data: RC extraction requires rest periods of at least
    :attr:`min_rest` seconds across a reasonable SOC range.

    Algorithm:
        1. Locate rest periods of at least :attr:`min_rest` seconds
        2. Assign an SOC to each measurement based on :attr:`capacity`
        3. Fit :attr:`n_rc` exponential models to the IV(t) data
        4. The scale and power parameters are converted to the RC parameters
        5. Fit a 1-D smoothing cubic spline for voltage as a function of SOC,
           placing knots at :attr:`soc_points`.
        6. Evaluate the spline at SOC points requested by the user,
           return as a :class:`~moirae.models.ecm.components.Resistance` object
           using the :attr:`interpolation_style` type of spline.

    Args:
        capacity: Best estimate for capacity of the cell (Amp-hours)
        starting_soc: Best estimate for SOC at the beginning of cycle
        soc_points: SOC points at which to extract R0 or (``int``) number of
        grid points.
        soc_requirement: Require that dataset samples at least this fraction
        of the capacity
        n_rc: Number of RC couples in the ECM
        min_rest: Minimum required rest duration in seconds
        min_dur_prev: Minimum duration (in seconds) of the step that precedes the rest(s). Defaults to 1 minute
        max_rest_I: Maximum current expected during a rest period (Amps)
    """

    capacity: float
    """Best estimate for capacity of the cell"""
    starting_soc: float
    """Best estimate for SOC at the beginning of cycle"""
    soc_points: np.ndarray
    """State of charge points at which to estimate the resistance"""
    soc_requirement: float
    """Require that dataset samples at least this fraction of the capacity"""
    n_rc: int
    """Number of RC couples in the ECM"""
    min_rest: float
    """Minimum required rest duration in seconds"""
    min_dur_prev: float
    """Minimum duration (in seconds) of the step that precedes the rest(s). Defaults to 1 minute"""
    max_rest_I: float
    """Maximum current expected during a rest period (Amps)"""

    def __init__(self,
                 capacity: float | MaxTheoreticalCapacity,
                 starting_soc: float = 0.0,
                 soc_points: np.ndarray | int = 11,
                 soc_requirement: float = 0.95,
                 n_rc: int = 1,
                 min_rest: float = 600,
                 min_dur_prev: float = 60,
                 max_rest_I: float = None):

        if isinstance(soc_points, int):
            soc_points = np.linspace(0, 1, soc_points)
        self.capacity = capacity.base_values[0, 0] if isinstance(capacity, MaxTheoreticalCapacity) else float(capacity)
        self.starting_soc = starting_soc
        self.soc_points = np.array(soc_points)
        self.soc_requirement = soc_requirement
        self.n_rc = n_rc
        self.min_rest = min_rest
        self.min_dur_prev = min_dur_prev
        if max_rest_I is None:
            max_rest_I = self.capacity / 100
        self.max_rest_I = max_rest_I
        self.interpolation_style = 'linear'

    def check_data(self, data: BatteryDataset):
        # Check if there is a raw_data table
        if 'raw_data' not in data.tables:
            raise ValueError('`raw_data` table is required')

        if data.tables.get('cycle_stats') is None or 'capacity_charge' not in data.tables['cycle_stats'].columns:
            CapacityPerCycle().add_summaries(data)

        # Check if the rests sample a sufficient soc range
        rests = self._extract_rests(data)
        rest_socs = np.array([item['soc'] for item in rests])
        sampled_soc = rest_socs.max() - rest_socs.min()
        if sampled_soc < self.soc_requirement:
            raise ValueError(f'Dataset rests must sample {self.soc_requirement * 100:.1f}% of SOC.'
                             f' Only sampled {sampled_soc * 100:.1f}%')

    def interpolate_rc(self, data: BatteryDataset) -> np.ndarray:
        """Fit then evaluate a smoothing spline which explains
        RC values as a function of SOC

        Args:
            cycle: Cycle to use for fitting the spline
        Returns:
            An estimate for all RC parameters at :attr:`soc_points`

        """
        rests = self._extract_rests(data)

        RCs = {'soc': []}
        for i_rc in range(self.n_rc):
            RCs[f'R{i_rc + 1}'] = []
            RCs[f'C{i_rc + 1}'] = []

        for i, rest in enumerate(rests):

            if self.n_rc > 0:

                RCs['soc'].append(rest['soc'])

                bounds = np.zeros((2 * self.n_rc, 2))
                bounds[:, 1] = 1
                bounds[-1, 1] = 1000

                res = differential_evolution(
                    self._error, bounds, popsize=120, vectorized=True,
                    updating='deferred',
                    args=(rest['step_data'], rest['indx_rest'],
                          rest['t_rest'], rest['state']))

                params_fit = res.x.copy()

                for i_rc in np.arange(self.n_rc - 1)[::-1]:
                    # Ti = Ti/Ti+1 * Ti+1
                    params_fit[2 * i_rc + 1] *= params_fit[2 * (i_rc + 1) + 1]

                # transform trace A/T parameters into RC parameters
                params_rc = params_fit.copy()
                for i_rc in range(self.n_rc):
                    # We need to compute the current that flows through the resistive element at the beginning of the
                    # rest period. For that, we need the time constant tau
                    tau = params_rc[2 * i_rc + 1]
                    # Compute current assuming previous step started with zero charge in capacitor
                    i_r_rc = compute_I_RCs(total_current=rest['Iprev']['current'],
                                           timestamps=rest['Iprev']['time'],
                                           tau_values=tau).item()
                    # Update parameters
                    resistance = params_rc[2 * i_rc] / abs(i_r_rc)  # R = A/Iprev
                    params_rc[2 * i_rc] = resistance
                    capacitance = tau / resistance  # C = T/R
                    params_rc[2 * i_rc + 1] = capacitance

                    RCs[f'R{i_rc + 1}'].append(params_rc[2 * i_rc])
                    RCs[f'C{i_rc + 1}'].append(params_rc[2 * i_rc + 1])

        splines_eval = []
        # Find the indices that sort the SOC values
        soc_idx = np.argsort(RCs['soc'], )
        for i_rc in range(self.n_rc):
            # Get the values at the lowest and largest SOC
            r_low = RCs[f'R{i_rc + 1}'][soc_idx[0]]
            c_low = RCs[f'C{i_rc + 1}'][soc_idx[0]]
            r_high = RCs[f'R{i_rc + 1}'][soc_idx[-1]]
            c_high = RCs[f'C{i_rc + 1}'][soc_idx[-1]]
            # Create interpolation
            Rint = interp1d(x=RCs['soc'], y=RCs[f'R{i_rc + 1}'], bounds_error=False, fill_value=(r_low, r_high))
            Cint = interp1d(x=RCs['soc'], y=RCs[f'C{i_rc + 1}'], bounds_error=False, fill_value=(c_low, c_high))

            splines_eval.append(
                (Rint(self.soc_points),
                 Cint(self.soc_points)))

        return splines_eval

    def _extract_rests(self, data: BatteryDataset) -> List[Dict]:
        """Extract the relevant time-series segments for fitting RC elements

        Args:
            cycle: Dataset containing the time series measurement
        Returns:
            A list of relevant rest segments with associated soc, rest indices
            rest times, and the state and average current of the previous step
        """
        if data.tables.get('cycle_stats') is None or 'capacity_charge' not in data.tables['cycle_stats'].columns:
            CapacityPerCycle().add_summaries(data)

        cycle = data.tables['raw_data']
        cycle = cycle.copy(deep=False)  # We are not editing the data
        if 'cycled_charge' not in cycle.columns:
            StateOfCharge().enhance(cycle)
        cycle['soc'] = self.starting_soc + cycle['cycled_charge'] / self.capacity  # Ensure data are [0, 1)

        if 'state' not in cycle.columns:
            AddState(rest_curr_threshold=self.max_rest_I).enhance(cycle)
        if 'step_index' not in cycle.columns:
            AddSteps().enhance(cycle)
        grp = cycle.groupby('step_index')
        step_data_prev = None

        rests = []
        socs = []
        for step, step_data in grp:

            Iavg = step_data['current'].mean()
            Istd = step_data['current'].std()
            if np.abs(Iavg) > self.max_rest_I or Istd > self.max_rest_I:
                step_data_prev = step_data.copy()
                continue

            soc = np.mean(step_data['soc'])

            if step_data_prev is not None:
                prev_time = step_data_prev['test_time'].to_numpy()
                # Make sure the previous step is long enough our what we want
                if prev_time[-1] - prev_time[0] >= self.min_dur_prev:
                    Iprev = {'current': step_data_prev['current'].to_numpy(),
                             'time': prev_time}
                    step_data = pd.concat(
                        [step_data_prev.iloc[-2:], step_data])
                else:
                    Iprev = {'current': np.nan, 'time': np.nan}
            else:
                Iprev = {'current': np.nan, 'time': np.nan}

            # find the state (charging or discharging) of the previous step
            if 'charging' in step_data['state'].values:
                sel = step_data['state'] == 'charging'
                state = 'ch'
            elif 'discharging' in step_data['state'].values:
                sel = step_data['state'] == 'discharging'
                state = 'di'
            else:
                step_data_prev = step_data.copy()
                continue

            step_data_chdi = step_data.loc[sel]

            if len(step_data_chdi) > 0:
                indx_end_chdi = step_data_chdi.index[-1]
            else:
                indx_end_chdi = step_data.index[0]

            tmp = (step_data['current'].loc[indx_end_chdi + 1:].abs()
                   > self.max_rest_I).cumsum() == 0
            indx_rest = tmp[tmp].index
            t_rest = step_data['test_time'][indx_rest]

            t_rest_el = t_rest.max() - t_rest.min()
            if t_rest_el < self.min_rest:
                step_data_prev = step_data.copy()
                continue

            socs.append(soc)

            rests.append({
                'soc': soc,
                'step_data': step_data,
                'indx_rest': indx_rest,
                't_rest': t_rest,
                'state': state,
                'Iprev': Iprev
            })

        rests_ = [rests[ii] for ii in np.argsort(socs)]

        return rests_

    def _error(self, params, cycle_data, indx_fitseg, t_fitseg, state):
        """Calculate the prediction MSE for the RC elements of a cell

        Args:
            params: The parameters of the RC pairs
            cycle_data: Dataset containing the time series measurement
            indx_fitseg: Time index for the fitting segment (the rest)
            t_fitseg: Time sequence for the fitting segment
            state: State of the previous segment (ch or di)
        Returns:
            An tuple of RC instances with the requested SOC interpolation points
        """
        params = params.T

        npset = params.shape[0]

        As = params[None, ..., 0::2]
        Ts = params[None, ..., 1::2]
        for i_rc in np.arange(self.n_rc - 1)[::-1]:
            Ts[..., i_rc] *= Ts[..., i_rc + 1]

        t = np.array(t_fitseg) - np.array(t_fitseg)[0]
        t = t[..., None]

        res = cycle_data['voltage'][indx_fitseg[-1]]
        res = np.tile(res, (t.size, npset))

        if state == 'di':
            for i_rc in range(self.n_rc):
                res -= As[..., i_rc] * np.exp(-t / Ts[..., i_rc])

        elif state == 'ch':
            for i_rc in range(self.n_rc):
                res += As[..., i_rc] * np.exp(-t / Ts[..., i_rc])

        ref = cycle_data['voltage'][indx_fitseg]
        ref = np.array(ref)[:, None]

        err = np.sum((ref - res) ** 2, axis=0)

        err[np.isnan(err)] = np.inf

        return err

    def extract(self, data: Union[pd.DataFrame, BatteryDataset]) -> Tuple[RCComponent, ...]:
        """Extract an estimate for the RC elements of a cell

        Args:
            dataset: Dataset containing time series measurements.

        Returns:
            An tuple of RC instances with the requested SOC interpolation points
        """
        # Ensure correct object
        dataset = ensure_battery_dataset(data=data)

        # knotsl: list of tuples of interpolated R and C values
        knotsl = self.interpolate_rc(dataset)

        RCcomps = tuple()
        for knots in knotsl:
            RC_R = Resistance(
                base_values=knots[0],
                soc_pinpoints=self.soc_points,
                interpolation_style=self.interpolation_style)
            RC_C = Capacitance(
                base_values=knots[1],
                soc_pinpoints=self.soc_points,
                interpolation_style=self.interpolation_style)
            RCcomps += (RCComponent(r=RC_R, c=RC_C).model_copy(),)

        return RCcomps
