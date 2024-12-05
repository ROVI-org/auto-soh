"""Extraction algorithms which gather parameters of an ECM"""
import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline

from battdat.data import CellDataset
from battdat.postprocess.integral import CapacityPerCycle
from moirae.extractors.base import BaseExtractor
from moirae.models.ecm.components import ReferenceOCV, OpenCircuitVoltage, MaxTheoreticalCapacity


class MaxCapacityExtractor(BaseExtractor):
    """Estimate the maximum capacity of a battery

    Suggested Data: Low current cycles which sample fully charge or discharge a battery

    Algorithm:
        1. Compute the observed capacity each cycle if not available
           using :class:`battdat.postprocess.integral.CapacityPerCycle`.
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
    the entire SOC range with a slow charge and discharge rate. Periodic rests are helpful
    but not required.

    Algorithm:
        1. Locate cycle with the lowest average voltage during charge and discharge
        2. Assign an SOC to each measurement based on the nominal capacity
        3. Assign a weights to each point based on :math:`1 / max(\\left| current \\right|, 1e-6)`.
           Normalize weights such that they sum to 1.
        4. Fit a 2-D smoothing spline for voltage as a function of SOC and current.
           Use a cubic spline for SOC dependence and a linear spline for current dependence,
           which approximates a series resistor. Weigh points according to the values assigned
           in Step 3 and target a weighted root mean squared error of :attr:`target_error`.
        5. Evaluate the 2-D spline at SOC points requested by the user.

    Args:
        soc_points: SOC points at which to extract OCV or (``int``) number of grid points.
        target_error: Target root mean squared error for the smoothing spline. Units: V
    """

    soc_points: np.ndarray
    """State of charge points at which to estimate the resistance"""
    target_error: float
    """Target error for smoothing spline. ``s`` of the spline will be
    equal to the square of this value."""
    capacity: float
    """Assumed capacity of the cell. Units: A-hr"""
    soc_requirement: float
    """Require that dataset samples at least this fraction of the capacity"""

    def __init__(self,
                 capacity: float | MaxTheoreticalCapacity,
                 soc_points: np.ndarray | int = 11,
                 target_error: float = 1e-2,
                 soc_requirement: float = 0.95):
        if isinstance(soc_points, int):
            soc_points = np.linspace(0, 1, soc_points)
        self.soc_points = np.array(soc_points)
        self.target_error = target_error
        self.capacity = capacity.base_values[0, 0] if isinstance(capacity, MaxTheoreticalCapacity) else float(capacity)
        self.soc_requirement = soc_requirement

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
            raise ValueError(f'Dataset must sample {self.soc_requirement*100:.1f}% of SOC.'
                             f' Only sampled {sampled_soc*100:.1f}%')

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

        # Assign weights according to current so that low-current values are more important
        w = 1. / np.clip(np.abs(cycle['current']), a_min=1e-6, a_max=None)
        w /= w.sum()

        # Evaluate the smoothing spline
        spline = SmoothBivariateSpline(
            cycle['soc'].values, cycle['current'].values, cycle['voltage'].values, w=w, ky=1, s=self.target_error ** 2
        )
        return spline.ev(self.soc_points, 0)

    def extract(self, dataset: CellDataset) -> OpenCircuitVoltage:
        """Extract an estimate for the OCV of a cell

        Args:
            dataset: Dataset containing an estimate for the nominal capacity and time series measurements.

        Returns:
            An OCV instance with the requested SOC interpolation points,
        """
        knots = self.interpolate_ocv(dataset.tables['raw_data'])
        return OpenCircuitVoltage(
            ocv_ref=ReferenceOCV(base_values=knots, soc_pinpoints=self.soc_points)
        )
