"""
Collection of utility functions and classes for the models
"""
from typing import Optional

from .base import DegradationModel, HealthVariable, InputQuantities, OutputQuantities, GeneralContainer


class NoDegradation(DegradationModel):
    """
    Class corresponding to no degradation, that is, the updated A-SOH is the same as the previous A-SOH.
    """
    def update_asoh(self,
                    previous_asoh: HealthVariable,
                    new_inputs: InputQuantities = None,
                    new_transients: Optional[GeneralContainer] = None,
                    new_measurements: Optional[OutputQuantities] = None) -> HealthVariable:
        return previous_asoh.model_copy(deep=True)
