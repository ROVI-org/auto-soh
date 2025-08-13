"""
Defines OCV extractor
"""
from typing import Union

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from battdat.data import BatteryDataset
from battdat.postprocess.integral import CapacityPerCycle, StateOfCharge

from moirae.estimators.offline.DataCheckers.utils import ensure_battery_dataset
from moirae.estimators.offline.extractors.base import BaseExtractor
from moirae.models.ecm.components import SOCInterpolatedHealth, OpenCircuitVoltage, MaxTheoreticalCapacity


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
        if 'cycled_charge' not in cycle.columns:
            StateOfCharge().enhance(cycle)
        cycle['soc'] = (cycle['cycled_charge'] - cycle['cycled_charge'].min()) / \
                       (cycle['cycled_charge'].max() - cycle['cycled_charge'].min())
        cycle = cycle.sort_values('soc')

        # Assign weights according to current so that low-current values are more important
        w = 1. / np.clip(np.abs(cycle['current']), a_min=1e-6, a_max=None)
        w /= w.sum()

        # Fit then evaluate a monotonic function
        model = IsotonicRegression(out_of_bounds='clip').fit(cycle['soc'], cycle['voltage'], sample_weight=w)
        return model.predict(self.soc_points)

    def extract(self, data: Union[pd.DataFrame, BatteryDataset]) -> OpenCircuitVoltage:
        """Extract an estimate for the OCV of a cell

        Args:
            dataset: Dataset containing an estimate for the nominal capacity and time series measurements.

        Returns:
            An OCV instance with the requested SOC interpolation points,
        """
        # Ensure correct object
        dataset = ensure_battery_dataset(data=data)

        knots = self.interpolate_ocv(dataset.tables['raw_data'])
        return OpenCircuitVoltage(
            ocv_ref=SOCInterpolatedHealth(base_values=knots, soc_pinpoints=self.soc_points,
                                          interpolation_style=self.interpolation_style)
        )
