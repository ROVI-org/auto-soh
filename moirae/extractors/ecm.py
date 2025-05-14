"""Extraction algorithms which gather parameters of an ECM"""
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import differential_evolution
from battdat.data import BatteryDataset
from battdat.postprocess.integral import CapacityPerCycle, StateOfCharge
from battdat.postprocess.tagging import AddSteps, AddState

from moirae.extractors.base import BaseExtractor
from moirae.models.ecm.components import SOCInterpolatedHealth, OpenCircuitVoltage, MaxTheoreticalCapacity
from moirae.models.ecm.components import Resistance, Capacitance, RCComponent


class MaxCapacityExtractor(BaseExtractor):
    """Estimate the maximum discharge capacity of a battery

    Suggested Data: Low current cycles which sample fully charge or discharge a battery

    Algorithm:
        1. Compute the observed capacity each cycle if not available
           using :class:`~battdat.postprocess.integral.CapacityPerCycle`.
        2. Find the maximum capacity over all provided cycles
    """

    def extract(self, data: BatteryDataset) -> MaxTheoreticalCapacity:
        # Access or compute cycle-level capacity
        cycle_stats = data.tables.get('cycle_stats')
        if cycle_stats is None or 'capacity_charge' not in cycle_stats:
            cycle_stats = CapacityPerCycle().compute_features(data)

        max_q = cycle_stats['capacity_charge'].max()
        return MaxTheoreticalCapacity(base_values=max_q)


class OCVExtractor(BaseExtractor):
    """Estimate the Open Circuit Voltage (OCV) of a battery as a function of state of charge (SOC)

    Suggested data: OCV extraction works best when provided with data for a cycle that samples
    the entire SOC range (at least a range larger than :attr:`soc_requirement`)
    with a slow charge and discharge rate. Periodic rests are helpful but not required.

    Algorithm:
        1. Locate cycle with the lowest average voltage during charge and discharge
        2. Assign an SOC to each measurement based on :attr:`capacity`
        3. Assign a weights to each point based on :math:`1 / max(\\left| current \\right|, 1e-6)`.
           Normalize weights such that they sum to 1.
        4. Fit an `isotonic regressor <https://scikit-learn.org/stable/modules/isotonic.html#isotonic>`_
           to the weighted data.
        5. Evaluate the regression at SOC points requested by the user,
           return as a :class:`~moirae.models.ecm.components.OpenCircuitVoltage` object
           using the :attr:`interpolation_style` type of spline.

    Args:
        soc_points: SOC points at which to extract OCV or (``int``) number of grid points.
    """

    soc_points: np.ndarray
    """State of charge points at which to estimate the resistance"""
    capacity: float
    """Assumed capacity of the cell. Units: A-hr"""
    soc_requirement: float
    """Require that dataset samples at least this fraction of the capacity"""
    interpolation_style: str
    """Type of spline used for the output"""

    def __init__(self,
                 capacity: float | MaxTheoreticalCapacity,
                 soc_points: np.ndarray | int = 11,
                 soc_requirement: float = 0.95,
                 interpolation_style: str = 'linear'):
        if isinstance(soc_points, int):
            soc_points = np.linspace(0, 1, soc_points)
        self.soc_points = np.array(soc_points)
        self.capacity = capacity.base_values[0, 0] if isinstance(capacity, MaxTheoreticalCapacity) else float(capacity)
        self.soc_requirement = soc_requirement
        self.interpolation_style = interpolation_style

    def check_data(self, data: BatteryDataset):
        # Check if there is a raw_data table
        if 'raw_data' not in data.tables:
            raise ValueError('`raw_data` table is required')

        # Compute the per-cycle capacity if unavailable
        if data.tables.get('cycle_stats') is None or 'capacity_charge' not in data.tables['cycle_stats'].columns:
            CapacityPerCycle().add_summaries(data)

        # Ensure at least one cycle samples capacities within
        sampled_soc = data.tables['cycle_stats']['capacity_charge'].max() / self.capacity
        if sampled_soc < self.soc_requirement:
            raise ValueError(f'Dataset must sample {self.soc_requirement * 100:.1f}% of SOC.'
                             f' Only sampled {sampled_soc * 100:.1f}%')

    def interpolate_ocv(self, cycle: pd.DataFrame) -> np.ndarray:
        """Fit then evaluate a smoothing spline which explains voltage as a function of SOC and current

        Args:
            cycle: Cycle to use for fitting the spline
        Returns:
            An estimate for the OCV at :attr:`soc_points`
        """
        # Compute the SOC by assuming the cycle fully discharges the batter
        #  TODO (wardlt): Make whether the cell started as charged (SOC~1) an option
        #   This code assumes the cycle starts with a discharged cell
        cycle = cycle.copy(deep=False)  # We are not editing the data
        if 'cycle_capacity' not in cycle.columns:
            StateOfCharge().enhance(cycle)
        cycle['soc'] = (cycle['cycle_capacity'] - cycle['cycle_capacity'].min()) / \
                       (cycle['cycle_capacity'].max() - cycle['cycle_capacity'].min())
        cycle = cycle.sort_values('soc')

        # Assign weights according to current so that low-current values are more important
        w = 1. / np.clip(np.abs(cycle['current']), a_min=1e-6, a_max=None)
        w /= w.sum()

        # Fit then evaluate a monotonic function
        model = IsotonicRegression(out_of_bounds='clip').fit(cycle['soc'], cycle['voltage'], sample_weight=w)
        return model.predict(self.soc_points)

    def extract(self, dataset: BatteryDataset) -> OpenCircuitVoltage:
        """Extract an estimate for the OCV of a cell

        Args:
            dataset: Dataset containing an estimate for the nominal capacity and time series measurements.

        Returns:
            An OCV instance with the requested SOC interpolation points,
        """
        knots = self.interpolate_ocv(dataset.tables['raw_data'])
        return OpenCircuitVoltage(
            ocv_ref=SOCInterpolatedHealth(base_values=knots, soc_pinpoints=self.soc_points,
                                          interpolation_style=self.interpolation_style)
        )


class R0Extractor(BaseExtractor):
    """Estimate the Instantaneous Resistance (R0) of a battery as a function of
    state of charge (SOC)

    Suggested data: R0 extraction works best when provided with data for a
    cycle that samples instantaneous changes in current across the entire SOC
    range (at least a range larger than :attr:`soc_requirement`)

    Algorithm:
        1. Locate cycle with jumps in current across SOC range
        2. Assign an SOC to each measurement based on :attr:`capacity`
        3. Calculate instantaneous resistance as dI/dt
        4. Filter for R0 values with dt below the threshold specified by
           :attr:`dt_max` and dI above the threshold specified by
           :attr:`dInorm_min`
        5. Fit a 1-D smoothing cubic spline for voltage as a function of SOC,
           placing knots at :attr:`soc_points`.
        6. Evaluate the spline at SOC points requested by the user,
           return as a :class:`~moirae.models.ecm.components.Resistance` object
           using the :attr:`interpolation_style` type of spline.

    Args:
        soc_points: SOC points at which to extract R0 or (``int``) number of
        grid points.
    """

    soc_points: np.ndarray
    """State of charge points at which to estimate the resistance"""
    soc_requirement: float
    """Require that dataset samples at least this fraction of the capacity"""
    dt_max: float
    """Max timestep for valid R0 instance"""
    dInorm_min: float
    """Min normalized current change for valid R0"""

    def __init__(self,
                 capacity: float | MaxTheoreticalCapacity,
                 soc_points: np.ndarray | int = 11,
                 soc_requirement: float = 0.95,
                 dt_max: float = 0.02,
                 dInorm_min: float = 0.1):

        if isinstance(soc_points, int):
            soc_points = np.linspace(0, 1, soc_points)
        self.capacity = capacity.base_values[0, 0] if isinstance(capacity, MaxTheoreticalCapacity) else float(capacity)
        self.soc_points = np.array(soc_points)
        self.soc_requirement = soc_requirement
        self.dt_max = dt_max
        self.dInorm_min = dInorm_min
        self.interpolation_style = 'linear'

    def check_data(self, data: BatteryDataset):
        # Check if there is a raw_data table
        if 'raw_data' not in data.tables:
            raise ValueError('`raw_data` table is required')

        if data.tables.get('cycle_stats') is None or 'capacity_charge' not in data.tables['cycle_stats'].columns:
            CapacityPerCycle().add_summaries(data)

        # Ensure at least one cycle samples capacities within
        sampled_soc = data.tables['cycle_stats']['capacity_charge'].max() / self.capacity
        if sampled_soc < self.soc_requirement:
            raise ValueError(f'Dataset must sample {self.soc_requirement * 100:.1f}% of SOC.'
                             f' Only sampled {sampled_soc * 100:.1f}%')

    def interpolate_r0(self, cycle: pd.DataFrame) -> np.ndarray:
        """Fit then evaluate a smoothing spline which explains
        R0 as a function of SOC

        Args:
            cycle: Cycle to use for fitting the spline
        Returns:
            An estimate for the R0 at :attr:`soc_points`

        """
        # calculate soc throughout cycle
        cycle = cycle.copy(deep=False)
        if 'cycle_capacity' not in cycle.columns:
            StateOfCharge().enhance(cycle)
        cycle['soc'] = (cycle['cycle_capacity'] - cycle['cycle_capacity'].min()) / \
                       (cycle['cycle_capacity'].max() - cycle['cycle_capacity'].min())

        # calculate instantanous resistance at all points
        cycle['r0_inst'] = np.abs(
            cycle['voltage'].diff() /
            cycle['current'].diff())

        # calculate key quantities to filter R0_inst
        Inorm = cycle['cycle_capacity'].max() - cycle['cycle_capacity'].min()
        cycle['dInorm'] = cycle['current'].diff() / Inorm
        cycle['dt'] = cycle['test_time'].diff()

        # select instantaneous resistance values with
        # small timesteps and large changes in current
        sel_t = cycle['dt'] <= self.dt_max
        sel_I = np.abs(cycle['dInorm']) >= self.dInorm_min
        top_r0 = cycle[sel_t * sel_I]
        top_r0 = top_r0.replace([np.inf, -np.inf], np.nan).dropna()
        top_r0 = top_r0.sort_values('soc')

        # Calculate absolute differences with next row
        tol = 0.01
        diffs = top_r0['soc'].diff().abs() <= tol
        top_r0 = top_r0[~diffs]

        # Evaluate the smoothing spline
        x_data = top_r0['soc'].values
        y_data = top_r0['r0_inst'].values
        t = self.soc_points[1:-1]

        spline = LSQUnivariateSpline(x_data, y_data, t=t, k=1)
        return spline(self.soc_points)

    def extract(self, dataset: BatteryDataset) -> Resistance:
        """Extract an estimate for the R0 of a cell

        Args:
            dataset: Dataset containing time series measurements.

        Returns:
            An R0 instance with the requested SOC interpolation points,
        """

        # Compute the capacity integrates if not available
        knots = self.interpolate_r0(dataset.tables['raw_data'])
        return Resistance(base_values=knots, soc_pinpoints=self.soc_points,
                          interpolation_style=self.interpolation_style)


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
        soc_points: SOC points at which to extract R0 or (``int``) number of
        grid points.
        soc_requirement: Require that dataset samples at least this fraction
        of the capacity
        n_rc: Number of RC couples in the ECM
        min_rest: Minimum required rest duration in seconds
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
    max_rest_I: float
    """Maximum current expected during a rest period (Amps)"""

    def __init__(self,
                 capacity: float | MaxTheoreticalCapacity,
                 starting_soc: float = 0.0,
                 soc_points: np.ndarray | int = 11,
                 soc_requirement: float = 0.95,
                 n_rc: int = 1,
                 min_rest: float = 600,
                 max_rest_I: float = None):

        if isinstance(soc_points, int):
            soc_points = np.linspace(0, 1, soc_points)
        self.capacity = capacity.base_values[0, 0] if isinstance(capacity, MaxTheoreticalCapacity) else float(capacity)
        self.starting_soc = starting_soc
        self.soc_points = np.array(soc_points)
        self.soc_requirement = soc_requirement
        self.n_rc = n_rc
        self.min_rest = min_rest
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

        for rest in rests:

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
                    params_rc[2 * i_rc] /= rest['Iprev']  # R = A/Iprev
                    params_rc[2 * i_rc + 1] /= params_fit[2 * i_rc]  # C = T/R

                    RCs[f'R{i_rc + 1}'].append(params_rc[2 * i_rc])
                    RCs[f'C{i_rc + 1}'].append(params_rc[2 * i_rc + 1])

        # Evaluate the smoothing spline for each element of each RC pair
        t = self.soc_points[1:-1]

        splines_eval = []
        for i_rc in range(self.n_rc):
            Rspl = LSQUnivariateSpline(RCs['soc'], RCs[f'R{i_rc + 1}'], t=t, k=1)
            Cspl = LSQUnivariateSpline(RCs['soc'], RCs[f'C{i_rc + 1}'], t=t, k=1)

            splines_eval.append(
                (Rspl(self.soc_points),
                 Cspl(self.soc_points)))

        return splines_eval

    def _extract_rests(self, data: BatteryDataset) -> np.ndarray:
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
        if 'cycle_capacity' not in cycle.columns:
            StateOfCharge().enhance(cycle)
        cycle['soc'] = self.starting_soc + cycle['cycle_capacity'] / self.capacity  # Ensure data are [0, 1)

        if 'state' not in cycle.columns:
            AddState().enhance(cycle)
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
                Iprev = np.abs(step_data_prev['current'].mean())
                step_data = pd.concat(
                    [step_data_prev.iloc[-2:], step_data])
            else:
                Iprev = np.nan

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
                   > 1e-5).cumsum() == 0
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

    def extract(self, dataset: BatteryDataset) -> Tuple[RCComponent, ...]:
        """Extract an estimate for the RC elements of a cell

        Args:
            dataset: Dataset containing time series measurements.

        Returns:
            An tuple of RC instances with the requested SOC interpolation points
        """

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
