"""Representations for the parameters of a Thenevin model"""
import numpy as np
from typing import Tuple

from pydantic import Field, model_validator, ConfigDict

from moirae.models.base import HealthVariable, ScalarParameter, GeneralContainer, ListParameter
from moirae.models.components.soc import (
    SOCDependentHealth, SOCPolynomialHealth
)
from moirae.models.components.soc_t import (
    SOCTempDependentHealth, SOCTempPolynomialHealth
)


class TheveninASOH(HealthVariable):
    """Parameters which describe the parameters of a Thevenin circuit

    These parameters match the parameters required to build a :class:`thevenin.Model`.
    """
    model_config = ConfigDict(use_attribute_docstrings=True)

    # Default parameters from: test_model.py in thevenin
    capacity: ScalarParameter = 1.
    """Maximum battery capacity (A-hr)"""
    mass: ScalarParameter = 0.1
    """Total battery mass of a battery (kg)"""
    c_p: ScalarParameter = 1150.
    """Specific heat capacity (J/kg/K)"""
    h_thermal: ScalarParameter = 12
    """Convective coefficient (W/m^2/K)"""
    a_therm: ScalarParameter = 1
    """Heat loss area (m^2)"""
    ocv: SOCDependentHealth = Field(default_factory=lambda: SOCPolynomialHealth(coeffs=3.5))
    """Open circuit voltage (V)"""
    r: Tuple[SOCTempDependentHealth, ...] = Field(
        default_factory=lambda: (SOCTempPolynomialHealth(soc_coeffs=0.1),), min_length=1
    )
    """Resistance all resistors, including both the series resistor and those in RC elements (Ohm)"""
    c: Tuple[SOCTempDependentHealth, ...] = Field(default_factory=tuple)
    """Capacitance in all RC elements (F)"""
    ce: ScalarParameter = 1.
    """Coulomb efficiency"""
    gamma: ScalarParameter = 50.
    """Hysteresis approach rate"""
    m_hyst: SOCDependentHealth = Field(default_factory=lambda: SOCPolynomialHealth(coeffs=0))
    """Maximum magnitude of hysteresis (V)"""

    @property
    def num_rc_elements(self) -> int:
        return len(self.c)

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
    hyst: ScalarParameter = 0.
    """Hysteresis voltage (units: V)"""
    eta: ListParameter = ()
    """Overpotential for the RC elements (units: V)"""

    @classmethod
    def from_asoh(cls, asoh: TheveninASOH, soc: float = 0., temp: float = 298.) -> 'TheveninTransient':
        """
        Create a transient state appropriate for the circuit defined in a :class:`TheveninASOH`.

        Args:
            asoh: Circuit definition
            soc: Starting SOC
            temp: Starting temperature (units: K)

        Returns:
            A state with the desired SOC, temperature, and all RC elements fully discharged.
        """

        eta = np.zeros((1, asoh.num_rc_elements))
        return cls(soc=soc, temp=temp, eta=eta)
