"""Extraction algorithms which gather parameters of an ECM"""
import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline

from battdat.data import CellDataset
from moirae.models.ecm.components import ReferenceOCV, OpenCircuitVoltage


class OCVExtractor:
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

    def __init__(self, soc_points: np.ndarray | int = 11, target_error: float = 1e-2):
        if isinstance(soc_points, int):
            soc_points = np.linspace(0, 1, soc_points)
        self.soc_points = np.array(soc_points)
        self.target_error = target_error

    def find_best_cycle(self, dataset: CellDataset) -> pd.DataFrame:
        """Locate a cycle with the smallest maximum current

        Args:
            dataset: Dataset containing the raw measurements of a cell
        Returns:
            A subset from the dataframe with the smallest maximum current
        """
        raw_data = dataset.raw_data
        min_i = raw_data.groupby('cycle_number')['current'].agg(lambda x: np.abs(x).max()).idxmin()
        return raw_data.query(f'cycle_number == {min_i}')

    def interpolate_ocv(self, dataset: CellDataset, cycle: pd.DataFrame) -> np.ndarray:
        """Fit then evaluate a smoothing spline which explains voltage as a function of SOC and current

        Args:
            dataset: Dataset containing the battery metadata
            cycle: Cycle to use for fitting the spline
        Returns:
            An estimate for the OCV at :attr:`soc_points`
        """
        cap = dataset.metadata.battery.nominal_capacity
        cycle = cycle.copy(deep=False)  # We are not editing the data
        cycle['soc'] = cycle['cycle_capacity'] / cap  # Ensure data are [0, 1)

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
        cycle = self.find_best_cycle(dataset)
        knots = self.interpolate_ocv(dataset, cycle)
        return OpenCircuitVoltage(
            ocv_ref=ReferenceOCV(base_values=knots, soc_pinpoints=self.soc_points)
        )
