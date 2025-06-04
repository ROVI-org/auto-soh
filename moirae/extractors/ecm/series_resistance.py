"""
Defines functionality for R0 extraction
"""
import numpy as np
import pandas as pd

from scipy.interpolate import LSQUnivariateSpline
from battdat.data import BatteryDataset
from battdat.postprocess.integral import CapacityPerCycle, StateOfCharge

from moirae.extractors.base import BaseExtractor
from moirae.models.ecm.components import MaxTheoreticalCapacity
from moirae.models.ecm.components import Resistance


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
