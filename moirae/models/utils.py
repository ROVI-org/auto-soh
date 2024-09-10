"""
Collection of utility functions and classes for the models
"""
from .base import DegradationModel, HealthVariable


class DummyDegradation(DegradationModel):
    """
    Class corresponding to no degradation, that is, the updated A-SOH is the same as the previous A-SOH.
    """
    def update_asoh(self, previous_asoh: HealthVariable, *args, **kwargs) -> HealthVariable:
        return previous_asoh.model_copy(deep=True)
