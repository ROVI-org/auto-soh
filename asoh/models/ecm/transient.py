# General imports
from typing import Union, Optional, Sized, List
from numbers import Number
from pydantic import Field

# ASOH imports
from asoh.models.base import HiddenVector


################################################################################
#                                HIDDEN VECTOR                                 #
################################################################################
class ECMTransientVector(HiddenVector, validate_assignment=True):
    soc: float = Field(default=0.0, description='State of charge (SOC)')
    q0: Optional[Union[float, None]] = \
        Field(default=None,
              description='Charge in the series capacitor. Units: Coulomb')
    i_rc: Optional[Union[float, List]] = \
        Field(default=[],
              description='Currents through RC components. Units: Amp')
    hyst: float = Field(default=0, description='Hysteresis voltage. Units: V')


################################################################################
#                               PROVIDE TEMPLATE                               #
################################################################################
def provide_transient_template(
        has_C0: bool,
        num_RC: float,
        soc: float = 0.0,
        q0: float = 0.0,
        i_rc: Union[float, List] = None,
        hysteresis: float = 0.0,
        num_copies: int = 1
        ) -> Union[ECMTransientVector, List[ECMTransientVector]]:
    """
    Function to help provide ECMTransientVector template copies based on ECM
    description. Allows for initialization of specific variables if needed.
    """
    hidden = ECMTransientVector(soc=soc, hyst=hysteresis)
    if has_C0:
        hidden.q0 = q0
    if num_RC:
        if i_rc is None:
            i_rc = [0 for _ in range(num_RC)]
        elif isinstance(i_rc, Number):
            i_rc = [i_rc for _ in range(num_RC)]
        elif isinstance(i_rc, Sized):
            if len(i_rc) != num_RC:
                raise ValueError('Mismatch between number of RC currents '
                                 'provided and number of RC elements!')
        hidden.i_rc = i_rc

    if num_copies == 1:
        return hidden
    else:
        return [hidden.model_copy() for _ in range(num_copies)]
