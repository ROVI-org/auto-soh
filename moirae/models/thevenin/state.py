"""Representations for the parameters of a Thenevin model"""
from typing import List

from pydantic import Field, model_validator

from moirae.models.base import HealthVariable, ScalarParameter, GeneralContainer, ListParameter
from moirae.models.thevenin.components import (
    SOCDependentVariable, SOCPolynomialVariable, SOCTempDependentVariable, SOCTempPolynomialVariable
)


class TheveninASOH(HealthVariable):
    """Parameters which describe the parameters of a Thevenin circuit

    These parameters match the parameters required to build a :class:`thevenin.Model`.
    """

    # Default parameters from: test_model.py in thevenin
    capacity: ScalarParameter = 1.
    """Maximum battery capacity (units: A-hr)"""
    mass: ScalarParameter = 0.1
    """Total battery mass of a battery (units: kg)"""
    c_p: ScalarParameter = 1150.
    """Specific heat capacity (units: J/kg/K)"""
    h_thermal: ScalarParameter = 12
    """Convective coefficient (units: W/m^2/K)"""
    a_therm: ScalarParameter = 1
    """Heat loss area (units: m^2)"""
    ocv: SOCDependentVariable = Field(default_factory=lambda: SOCPolynomialVariable(coeffs=3.5))
    """Open circuit voltage (units: V)"""
    r: List[SOCTempDependentVariable] = Field(
        default_factory=lambda: [SOCTempPolynomialVariable(soc_coeffs=0.1)], min_length=1
    )
    """Resistance all resistors, including both the series resistor and those in RC elements (units: Ohm)"""
    c: List[SOCTempDependentVariable] = Field(default_factory=list)
    """Capacitance in all RC elements (units: C)"""

    @model_validator(mode='after')
    def _check_rc_elements(self) -> 'TheveninASOH':
        """Ensure the r and c lists are consistent"""
        if len(self.r) - 1 != len(self.c):
            raise ValueError(f'There should be one more R than C element. Found {len(self.r)} R, {len(self.c)} C.')
        return self


class TheveninTransient(GeneralContainer):
    """Transient state of the ECM circuit"""

    soc: ScalarParameter = 0.
    """State of charge for the battery system"""
    temp: ScalarParameter = 298.
    """Temperature of the battery (units: K)"""
    eta: ListParameter = []
    """Overpotential for the RC elements (units: V)"""
