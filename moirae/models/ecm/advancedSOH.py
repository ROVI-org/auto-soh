"""Definition of the state of health"""
from typing import Tuple, Optional, Union, List

from pydantic import Field
import numpy as np
from scipy.integrate import trapezoid

from moirae.models.base import HealthVariable, ScalarParameter
from .components import (MaxTheoreticalCapacity,
                         Resistance,
                         Capacitance,
                         RCComponent,
                         ReferenceOCV,
                         EntropicOCV,
                         OpenCircuitVoltage,
                         HysteresisParameters)
from .utils import realistic_fake_ocv


class ECMASOH(HealthVariable):
    """State of Health for an equivalent circuit model"""
    q_t: MaxTheoreticalCapacity = Field(description='Maximum theoretical discharge capacity (Qt).')
    ce: ScalarParameter = Field(default=1., description='Coulombic efficiency (CE)')
    ocv: OpenCircuitVoltage = Field(description='Open Circuit Voltage (OCV)')
    r0: Resistance = Field(description='Series Resistance (R0)')
    c0: Optional[Capacitance] = Field(default=None, description='Series Capacitance (C0)')
    rc_elements: Tuple[RCComponent, ...] = Field(default_factory=tuple, description='Tuple of RC components')
    h0: HysteresisParameters = Field(default=HysteresisParameters(base_values=0.0),
                                     description='Hysteresis component')

    def get_theoretical_energy(self,
                               soc_limits: Tuple[float, float] = (0., 1.),
                               temperature: Optional[float] = None) -> Union[float, np.ndarray]:
        """
        Function that computes the theoretical energy of the cell in Wh.

        Computes cell energy by integrating Open-Circuit Voltage (OCV) over the supplied state of charge (SOC) ranges.
        Assumes no energy loss, such as from the resistive or hysteresis elements.

        Args:
            soc_limits: minimum and maximum SOC limit to be used in the computation of the energy; defaults to (0, 1)
            temperature: value of temperature (in °C) to be used in the calculation of energy; defaults to None

        Returns:
            value(s) of maximum theoretical energy as a 2D array of shape (asoh_batch_size, 1)
        """
        # Get SOC values to be used in the integration
        soc_vals = np.linspace(min(soc_limits), max(soc_limits), 100)
        # Get corresponding OCV
        ocv_vals = self.ocv(soc=soc_vals, temp=temperature)  # shape (ocv_batch, 1, soc_dim)
        # Integrate
        energy = trapezoid(y=ocv_vals, x=soc_vals, axis=-1)  # integrate over the last axis, corresponding to soc_dim
        # Now, multiply by the charge
        return energy * self.q_t.amp_hour  # shape (asoh_batch, 1)

    @classmethod
    def provide_template(
            cls,
            has_C0: bool,
            num_RC: int,
            qt: float = 10.0,
            CE: float = 1.0,
            OCV: Union[float, np.ndarray, None] = None,
            R0: Union[float, np.ndarray] = 0.05,
            C0: Union[float, np.ndarray, None] = None,
            H0: Union[float, np.ndarray] = 0.05,
            RC: Union[List[Tuple[np.ndarray, ...]], None] = None,
    ) -> 'ECMASOH':
        """Create an ECM using a simple template

        Args:
            has_C0: Whether circuit includes a serial capacitor
            num_RC: How many RC elements are within the circuit
            qt: Maximum theoretical capacity. (Units: Amp-hr)
            CE: Coulombic efficiency
            OCV: Open circuit voltage at equally-spaced SOCs. (Units: V)
            R0: Series resistance. (Units: Ohm)
            C0: Series capacitance (Units: Farad)
            H0: Hysteresis value at equally-spaced SOCs (Units: V)
            RC: List of tuples of (resistance, capacitance) (Units: V)

        Returns:
            A set of parameters describing the entire circuit
        """
        # Start preparing the requirements
        qt = MaxTheoreticalCapacity(base_values=qt)
        # R0 prep
        R0 = Resistance(base_values=R0, temperature_dependence_factor=0.0025)
        # OCV prep
        OCVent = EntropicOCV(base_values=0.005)
        if OCV is None:
            socs = np.linspace(0, 1, 20)
            OCVref = ReferenceOCV(base_values=realistic_fake_ocv(socs),
                                  interpolation_style='cubic')
        else:
            OCVref = ReferenceOCV(base_values=OCV)
        OCV = OpenCircuitVoltage(ocv_ref=OCVref, ocv_ent=OCVent)
        # H0 prep
        H0 = HysteresisParameters(base_values=H0, gamma=0.9)

        # C0 prep
        c0 = None
        if has_C0:
            if C0 is None:
                # Make it so that it's impact is at most 10 mV
                C0 = qt.value / 0.01  # Recall it's stored in Amp-hour
            c0 = Capacitance(base_values=C0)

        # RC prep
        RCcomps = ()
        if num_RC:
            if RC is None:
                RC_R = Resistance(base_values=0.01,
                                  temperature_dependence_factor=0.0025)
                RC_C = Capacitance(base_values=2500)
                RCcomps = tuple(RCComponent(r=RC_R, c=RC_C).model_copy()
                                for _ in range(num_RC))
            else:
                if len(RC) != num_RC:
                    raise ValueError('Amount of RC information provided does not '
                                     'match number of RC elements specified!')
                RCcomps = tuple()
                for RC_info in RC:
                    R_info = RC_info[0]
                    C_info = RC_info[1]
                    RC_R = Resistance(base_values=R_info,
                                      temperature_dependence_factor=0.0025)
                    RC_C = Capacitance(base_values=C_info)
                    RCcomps += (RCComponent(r=RC_R, c=RC_C).model_copy(),)

        return ECMASOH(q_t=qt, ce=CE, ocv=OCV, r0=R0, h0=H0, c0=c0, rc_elements=RCcomps)
