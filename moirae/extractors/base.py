"""Interface for extractors"""
from battdat.data import BatteryDataset

from moirae.models.base import HealthVariable


class BaseExtractor:
    """Base class for tools which determine parameters from special cycles"""

    def check_data(self, data: BatteryDataset):
        """Verify whether data contains features needed for algorithm

        Args:
            data: Data to be evaluated
        Raises:
            (ValueError) If the dataset is missing critical information
        """
        pass  # Default: no constraints

    def extract(self, data: BatteryDataset) -> HealthVariable:
        """Determine parameters of a physics model from battery dataset

        Args:
            data: Data to use for parameter assessment
        Returns:
            Part of the parameter set for a model.
        """
        raise NotImplementedError()
