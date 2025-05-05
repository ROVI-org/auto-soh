"""Inputs and outputs from a Thevenin model"""
from moirae.models.base import InputQuantities, ScalarParameter


class TheveninInput(InputQuantities):
    """Inputs for the Thevenin model"""

    t_inf: ScalarParameter = 25.
    """Environmental temperature (units: °C)"""
