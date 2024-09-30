from typing import Union, Sized, Optional
from numbers import Number

from pydantic import Field
import numpy as np

from moirae.models.base import GeneralContainer, ScalarParameter, ListParameter
from .advancedSOH import ECMASOH


class ECMTransientVector(GeneralContainer):
    """Description of the state of charge of an ECM and all components"""

    soc: ScalarParameter = Field(0., description='SOC')
    q0: Optional[ScalarParameter] = Field(None, description='Charge in the series capacitor. Units: Coulomb')
    i_rc: ListParameter = Field(description='Currents through RC components. Units: Amp')
    hyst: ScalarParameter = Field(0., description='Hysteresis voltage. Units: V')

    @classmethod
    def provide_template(cls,
                         has_C0: bool,
                         num_RC: int,
                         soc: float = 0.0,
                         q0: float = 0.0,
                         i_rc: Union[float, np.ndarray, None] = None,
                         hysteresis: float = 0.0,
                         ) -> 'ECMTransientVector':
        """
        Function to help provide ECMTransientVector template copies based on ECM
        description. Allows for initialization of specific variables if needed.

        Args:
            has_C0: Whether circuit includes a serial capacitor
            num_RC: How many RC elements are within the circuit
            soc: State of charge between 0 and 1
            q0: Charge on the serial capacity (units: C)
            i_rc: Current over each of the rc_elements  (units: A)
            hysteresis: Hysteresis value (units: V)
        Returns:
            A set of parameters describing the current charge state
        """

        # Determine the starting current
        if i_rc is None:
            i_rc = np.zeros(num_RC)
        elif isinstance(i_rc, Number):
            i_rc = i_rc * np.ones(num_RC)
        elif isinstance(i_rc, Sized):
            if len(i_rc) != num_RC:
                raise ValueError('Mismatch between number of RC currents '
                                 'provided and number of RC elements!')
        i_rc = np.array(i_rc)

        return ECMTransientVector(soc=soc, hyst=hysteresis, i_rc=i_rc, q0=q0 if has_C0 else None)

    @classmethod
    def from_asoh(cls, asoh: ECMASOH):
        """
        Make a transient vector based on the design of a circuit captured in the state of health

        Args:
            asoh: State of health for an ECM

        Returns:
            An appropriate transient state
        """

        has_C0 = asoh.c0 is not None
        num_RC = len(asoh.rc_elements)
        return ECMTransientVector.provide_template(has_C0=has_C0, num_RC=num_RC)
