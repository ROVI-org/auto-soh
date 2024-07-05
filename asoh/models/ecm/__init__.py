"""Models that describe the state of health and transient state of an Equivalent Circuit Model"""
from typing import Tuple

from pydantic import Field

from asoh.models.base import HealthVariable
from .components import Resistor, OpenCircuitVoltage, RCElement


class ECMASOH(HealthVariable):
    """State of a health for battery defined by an equivalent circuit model"""

    r0: Resistor
    """Resistor in series with the battery"""
    rc_elements: Tuple[RCElement, ...]
    """A series of RC elements in series with the battery"""
    ocv: OpenCircuitVoltage = Field(description='Model for the open circuit voltage')
