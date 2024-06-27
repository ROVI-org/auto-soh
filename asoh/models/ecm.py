from typing import Any, Dict, List, Optional, Union, Literal
import numpy as np
from pydantic import Field, computed_field
from pydantic.fields import FieldInfo
from scipy.interpolate import interp1d

from .base import InputQuantities, OutputMeasurements, HealthParameter
from .base import AdvancedStateOfHealth


################################################################################
#                               INPUTS & OUTPUT                                #
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
#                                HEALTH METRICS                                #
################################################################################
class MaxTheoreticalCapacity(HealthParameter):
    """
    Defines maximum theoretical discharge capacity of a cell
    """
    Q_t: float = \
        Field(
            description='Maximum theoretical discharge capacity of a cell. Units: Amp-hour')
    updatable: bool = \
        Field(default=True,
              description='Define if external model can update this variable.')

    @property
    def value(self) -> float:
        return self.Q_t

    def to_numpy(self) -> np.array:
        """
        Returns maximum theoretical discharge capacity in Amp-seconds.
        """
        if self.updatable:
            return 3600 * np.array(self.Q_t)
        return np.array([])


class SeriesResistance(HealthParameter):
    """
    Defines the series resistance component of an ECM.
    """
    internal_parameters: Union[float, List, np.ndarray] = \
        Field(
            description='Values of series resistance at specified SOCs. Units: Ohm')
    soc_pinpoints: Optional[Union[List, np.ndarray]] = \
        Field(default=[], description='SOC pinpoints for interpolation.')
    updatable: bool = \
        Field(default=True,
              description='Define if external model can update this variable.')
    interpolation_style: \
        Literal['linear', 'nearest', 'nearest-up', 'zero', 'slinear',
                'quadratic', 'cubic', 'previous', 'next'] = \
        Field(default='linear', description='Type of interpolation to perform')
    reference_temperature: Optional[float] = \
        Field(default=25,
              description='Reference temperature for internal parameters. Units: °C')
    temperature_dependence_factor: Optional[float] = \
        Field(default=0,
              description='Factor determining dependence of R0 with temperature. Units: 1/°C')

    @computed_field
    @property
    def _interp_func(self) -> callable:
        """
        Interpolate values of R0. If soc_pinpoints have not been set, assume
        internal_parameters are evenly spread on an SOC interval [0,1].
        """
        if not len(self.soc_pinpoints):
            self.soc_pinpoints = np.linspace(0, 1, len(self.internal_parameters))
        func = interp1d(self.soc_pinpoints,
                        self.internal_parameters,
                        kind=self.interpolation_style,
                        bounds_error=False,
                        fill_value='extrapolate')
        return func

    def value(self, soc, temp=None) -> Union[float, np.ndarray]:
        """
        Computes value of series resistance at a given SOC and temperature.
        """
        reference_value = self._interp_func(soc)
        if temp is None or self.temperature_dependence_factor == 0:
            return reference_value
        gamma = self.temperature_dependence_factor
        deltaT = temp - self.reference_temperature
        new_value = reference_value * np.exp(- gamma * deltaT)
        return new_value

    def to_numpy(self) -> np.ndarray:
        if self.updatable:
            return self.internal_parameters
        return np.array([])


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
                                           default=f_value)

        cls.model_fields.update(new_fields)
        cls.model_rebuild(force=True)
