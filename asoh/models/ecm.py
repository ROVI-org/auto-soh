from typing import Any, Dict, Optional
import numpy as np
from pydantic import Field
from pydantic.fields import FieldInfo

from .base import InputQuantities, OutputMeasurements, HealthParameter
from .base import AdvancedStateOfHealth

################################################################################
##                              INPUTS & OUTPUT                               ##
################################################################################
class ECM_Input(InputQuantities):
    """
    Control of a battery based on the feed current, temperature
    """
    pass

# TODO (vventuri): Remeber we need to implement ways to denoise SOC, Qt, R0, 
#                   which require more outputs
class ECM_Measurement(OutputMeasurements):
    """
    Controls the outputs of the ECM.
    """
    pass

################################################################################
##                               HEALTH METRICS                               ##
################################################################################
class MaxTheoreticalCapacity(HealthParameter):
    """
    Defines maximum theoretical discharge capacity of a cell
    """
    Q_t: float = \
        Field(description = 'Maximum theoretical discharge capacity of a cell. Units: Amp-hour')
    updatable: bool = Field(default = True, description = '')
    
    def to_numpy(self) -> np.array:
        """
        Returns maximum theoretical discharge capacity in Amp-seconds
        """
        return 3600 * np.array(self.Q_t)
    

class SeriesResistance(HealthParameter):
    """
    Defines the series resistance component of an ECM.
    """
    pass



class ECM_ASOH(AdvancedStateOfHealth):
    pass

    def add_fields(cls, **field_definitions: Any):
        new_fields: Dict[str, FieldInfo] = {}
        new_annotations: Dict[str, Optional[type]] = {}

        for f_name, f_def in field_definitions.items():
            if isinstance(f_def, tuple):
                try:
                    f_annotation, f_value = f_def
                except ValueError as e:
                    raise Exception(
                        'field definitions should either be a tuple of (<type>, <default>) or just a '
                        'default value, unfortunately this means tuples as '
                        'default values are not allowed'
                    ) from e
            else:
                f_annotation, f_value = None, f_def

            if f_annotation:
                new_annotations[f_name] = f_annotation

            new_fields[f_name] = FieldInfo(annotation=f_annotation, 
                                           default = f_value)

        cls.model_fields.update(new_fields)
        cls.model_rebuild(force=True)


################################################################################

class ECMState(SystemState):
    """State of a battery defined by an Equivalent circuit model"""

    charge: float = Field(0, description='State of charge of the battery element. Units: A-hr')
    r_serial: float = Field(description='Resistance of resistor in series with the battery element', gt=0)
    ocv_params: tuple[float, float] = Field(description='Parameters which define the open-circuit voltage of the battery element. '
                                                        'Constant component (units: V), component which varies linearly with charge (units: V/A-hr)')

    state_params: tuple[str, ...] = ('charge',)

    def compute_ocv(self) -> float:
        """Compute the open circuit voltage (OCV) given at the current state of charge

        Returns:
            OCV in Volts
        """
        return self.ocv_params[0] + self.charge * self.ocv_params[1]


class ECMMeasurements(Measurements):
    """The only observable from an ECM model is the terminal voltage"""

    terminal_voltage: float = Field(description='Voltage at the terminal')


class SingleResistorModel(HealthModel):
    """A battery system modeled by a single resistor and open-circuit voltage which depends only on state of charge."""

    num_outputs = 1

    def dx(self, state: ECMState, control: ECMInput) -> np.ndarray:
        # The only change in the system is the state of charge increasing by the current
        return np.array([control.current / 3600.])

    def output(self, state: ECMState, control: InputState) -> ECMMeasurements:
        return ECMMeasurements(
            terminal_voltage=state.compute_ocv() + state.r_serial * control.current
        )
