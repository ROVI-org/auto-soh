"""Extraction algorithms which gather parameters of an ECM"""
import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline

from battdat.data import CellDataset
from battdat.postprocess.integral import CapacityPerCycle
from moirae.extractors.base import BaseExtractor
from moirae.models.ecm.components import ReferenceOCV, OpenCircuitVoltage, MaxTheoreticalCapacity


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
        4. Fit a 1-D smoothing cubic spline for voltage as a function of SOC,
           placing knots at :attr:`soc_points`.
        5. Evaluate the spline at SOC points requested by the user,
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
                 interpolation_style: str = 'cubic'):
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
        cycle = cycle.sort_values('soc')

        # Assign weights according to current so that low-current values are more important
        w = 1. / np.clip(np.abs(cycle['current']), a_min=1e-6, a_max=None)
        w /= w.sum()

        # Evaluate the smoothing spline
        t = self.soc_points[1:-1]
        spline = LSQUnivariateSpline(
            cycle['soc'].values, cycle['voltage'].values, w=w, t=t
        )
        return spline(self.soc_points)

    def extract(self, dataset: CellDataset) -> OpenCircuitVoltage:
        """Extract an estimate for the OCV of a cell

        Args:
            dataset: Dataset containing an estimate for the nominal capacity and time series measurements.

        Returns:
            An OCV instance with the requested SOC interpolation points,
        """
        knots = self.interpolate_ocv(dataset.tables['raw_data'])
        return OpenCircuitVoltage(
            ocv_ref=ReferenceOCV(base_values=knots, soc_pinpoints=self.soc_points,
                                 interpolation_style=self.interpolation_style)
        )
