from typing import Union, Optional, Sized
from numbers import Number

from pydantic import Field
import numpy as np

from moirae.models.base import TransientVector


class ECMTransientVector(TransientVector):
    """Description of the state of charge of an ECM and all components"""

    soc: float = Field(default=0.0, description='State of charge (SOC)')
    q0: Optional[float] = \
        Field(default=None,
              description='Charge in the series capacitor. Units: Coulomb')
    i_rc: Optional[np.ndarray] = \
        Field(default=None,
              description='Currents through RC components. Units: Amp')
    hyst: float = Field(default=0, description='Hysteresis voltage. Units: V')

    @classmethod
    def provide_template(cls,
                         has_C0: bool,
                         num_RC: int,
                         soc: float = 0.0,
                         q0: float = 0.0,
                         i_rc: Union[float, np.ndarray] = None,
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
        hidden = ECMTransientVector(soc=soc, hyst=hysteresis)
        if has_C0:
            hidden.q0 = q0
        if num_RC:
            if i_rc is None:
                i_rc = np.zeros(num_RC)
            elif isinstance(i_rc, Number):
                i_rc = i_rc * np.ones(num_RC)
            elif isinstance(i_rc, Sized):
                if len(i_rc) != num_RC:
                    raise ValueError('Mismatch between number of RC currents '
                                     'provided and number of RC elements!')
            hidden.i_rc = np.array(i_rc)

        return hidden
