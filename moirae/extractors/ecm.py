"""Extraction algorithms which gather parameters of an ECM"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import LSQUnivariateSpline
from battdat.data import CellDataset
from battdat.postprocess.integral import CapacityPerCycle


from moirae.extractors.base import BaseExtractor
from moirae.models.ecm.components import SOCInterpolatedHealth, OpenCircuitVoltage, MaxTheoreticalCapacity
from moirae.models.ecm.components import Resistance


class MaxCapacityExtractor(BaseExtractor):
    """Estimate the maximum discharge capacity of a battery

    Suggested Data: Low current cycles which sample fully charge or discharge a battery

    Algorithm:
        1. Compute the observed capacity each cycle if not available
           using :class:`~battdat.postprocess.integral.CapacityPerCycle`.
        2. Find the maximum capacity over all provided cycles
    """

    def extract(self, data: CellDataset) -> MaxTheoreticalCapacity:
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

    def check_data(self, data: CellDataset):
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
        cycle['soc'] = cycle['cycle_capacity'] / self.capacity  # Ensure data are [0, 1)
        cycle = cycle.sort_values('soc')

        # Assign weights according to current so that low-current values are more important
        w = 1. / np.clip(np.abs(cycle['current']), a_min=1e-6, a_max=None)
        w /= w.sum()

        # Fit then evaluate a monotonic function
        model = IsotonicRegression(out_of_bounds='clip').fit(cycle['soc'], cycle['voltage'], sample_weight=w)
        return model.predict(self.soc_points)

    def extract(self, dataset: CellDataset) -> OpenCircuitVoltage:
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
        3. Calculate instantanous resistance as dI/dt
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

    def check_data(self, data: CellDataset):
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
        cycle = self.soc_calc(cycle)

        # calculate instantanous resistance at all points
        cycle['r0_inst'] = np.abs(
            cycle['voltage'].diff() /
            cycle['current'].diff())

        # calculate key quantities to filter R0_inst
        Inorm = cycle['capacity'].max()
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
        self.top_r0 = top_r0

        # Evaluate the smoothing spline
        x_data = top_r0['soc'].values
        y_data = top_r0['r0_inst'].values
        t = self.soc_points[1:-1]

        spline = LSQUnivariateSpline(x_data, y_data, t=t, k=1)
        self.spline = spline

        return spline(self.soc_points)

    def plot_r0(self, n_plt=100):
        """Plot R0 vs SOC with Hermite interpolation"""
        x_plt = np.linspace(0, 1, n_plt)
        y_plt = self.spline(x_plt[:, None])

        plt.figure(num='instantaneous R0', figsize=(5, 4))

        plt.plot(x_plt, y_plt,
                 c='k', label='prediction')

        plt.scatter(self.top_R0['soc'], self.top_R0['R0_inst'],
                    c=self.top_R0['dt'], cmap='viridis')

        plt.xlabel('SOC')
        plt.ylabel('Instantaneous R0 (Ohms)')
        plt.legend()
        plt.colorbar(label='dt (s)')
        plt.tight_layout()
        plt.savefig("R0_spline.png")
        plt.close()

    def extract(self, dataset: CellDataset) -> Resistance:
        """Extract an estimate for the R0 of a cell

        Args:
            dataset: Dataset containing time series measurements.

        Returns:
            An R0 instance with the requested SOC interpolation points,
        """
        knots = self.interpolate_r0(dataset.tables['raw_data'])

        return Resistance(base_values=knots, soc_pinpoints=self.soc_points,
                          interpolation_style=self.interpolation_style)

    def soc_calc(self, cycle):
        """Compute accumulated capacity
        and state of charge"""

        capacity_ = np.array([])
        p = 0
        for key, s in cycle.groupby('step_index'):
            state = s['state'].iloc[0]
            if state == 'charging' or state == 'discharging':
                s_cap = cumulative_trapezoid(
                    s['current'],
                    s['test_time'],
                    initial=0) / 3600 + p
                p = s_cap[-1]
            else:
                s_cap = np.ones(len(s)) * p
                p = s_cap[-1]

            capacity_ = np.append(capacity_, s_cap)

        capacity_ -= capacity_[0]
        soc_ = capacity_ / capacity_.max()

        cycle['capacity'] = capacity_
        cycle['soc'] = soc_

        return cycle
