"""Base classes which define the state of a storage system,
the control signals applied to it, the outputs observable from it,
and the mathematical model which links state, control, and outputs together."""

from typing import Union, Optional, Literal, List
from abc import abstractmethod
from warnings import warn

import numpy as np
from pydantic import BaseModel, Field, computed_field, validate_call, ConfigDict


class GeneralContainer(BaseModel):
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.model_fields.keys())

    @property
    def all_names(self) -> tuple[str, ...]:
        all_names = self.names
        if ('extra', 'allow') in self.model_config.items():
            all_names += tuple(self.model_extra.keys())
        return all_names


class InputQuantities(GeneralContainer,
                      arbitrary_types_allowed=True,
                      validate_assignment=True):
    """
    Inputs quantities to a battery model, such as time, current, and temperature
    readings.
    """
    time: float = Field(description='Timestamp(s) of inputs. Units: s')
    current: float = Field(
        description='Current(s) applied to the storage system. Units: A')
    temperature: Optional[float] = \
        Field(default='Not provided!',
              description='Temperature reading(s). Units: °C')

    def to_numpy(self,
                 additional_inputs: tuple[str, ...] = ()) -> np.ndarray:
        """
        Outputs a numpy.ndarray where each entry is a np.ndarray corresponding
        to one input vector. This input vector ALWAYS has as its first two
        elements [time, current]. If provided, 'temperature' is the third
        element. Other additional inputs are appear in the order provided
        """
        combined = []
        for field in self.names:
            value = getattr(self, field)
            # since default for temperature is a string, we can hack it out
            if not isinstance(value, str):
                combined.append(value)
        combined += [getattr(self, add_in) for add_in in additional_inputs]

        return np.array(combined)


class OutputMeasurements(GeneralContainer,
                         arbitrary_types_allowed=True,
                         validate_assignment=True):
    """
    Ouput measurement of a battery model. Must include terminal voltage.
    """
    terminal_voltage: float = \
        Field(description='Voltage output of a battery cell/model. Units: V')

    def to_numpy(self,
                 additional_outputs: tuple[str, ...] = ()) -> np.ndarray:
        combined = [getattr(self, field) for field in self.names]
        combined += [getattr(self, add_out) for add_out in additional_outputs]
        return np.array(combined)


class HiddenVector(GeneralContainer,
                   arbitrary_types_allowed=True,
                   validate_assignment=True):
    """
    Holds the physical transient hidden state quantities (example: SOC, etc.)
    """
    def to_numpy(self) -> np.ndarray:
        transient_state = tuple(getattr(self, hid_var)
                                for hid_var in self.names)
        return np.hstack(transient_state)


class HealthVariable(BaseModel,
                     arbitrary_types_allowed=True,
                     validate_assignment=True):
    """
    Base definition for a health parameter, such as Q_total, R_0, etc.
    """
    base_values: Union[float, List] = \
        Field(
            description='Values used to parametrize health metric.')
    updatable: Union[Literal[False], tuple[str, ...]] = \
        Field(default=('base_values',),
              description='Define updatable parameters (if any)')

    def _get_internal_len(self, parameter_name: str) -> int:
        """
        Function to calculate length of a given parameter
        """
        param = getattr(self, parameter_name)
        if hasattr(param, '__len__'):
            return len(param)
        return 1

    @computed_field
    @property
    def updatable_len(self) -> int:
        """
        Denotes total length of a list containing all updatable parameters
        """
        if not self.updatable:
            return 0
        total_len = 0
        for internal_name in self.updatable:
            total_len += self._get_internal_len(internal_name)
        return total_len

    # TODO (vventuri): we should write a validator for the updatable field, to
    #                   make sure the parameters listed are all numbers.Number
    #                   or iterables composed of numbers.Number-s

    def get_updatable_parameter_values(self) -> List:
        """
        Function to obtain parameters used internally for health variable
        definition
        """
        all_params = []
        if not self.updatable:
            return all_params

        for internal_param in self.updatable:
            param = getattr(self, internal_param)
            if hasattr(param, '__len__'):
                all_params += list(param)
            else:
                all_params.append(param)
        return all_params

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def update(self,
               new_values: Union[float, List, np.ndarray],
               parameters: Union[tuple[str, ...], None] = None) -> None:
        """
        Method used to update updatable internal parameters with a new set of
        provided values.
        """
        if not self.updatable:
            warn('Attempt to update HealthVariable with no updatable parameters!'),
            return
        if parameters is None:
            parameters = self.updatable
        if isinstance(new_values, float):
            for param_name in parameters:
                if param_name in self.updatable:
                    setattr(self, param_name, new_values)
                else:
                    msg = 'Attempted to set \'' + param_name + '\', but '
                    msg += 'updatable parameters are ' + str(self.updatable)
                    msg += '! Skipping this one...'
                    warn(msg)
            return
        # Set index counter to determine where we need to read things from
        begin_id = 0
        for param_name in parameters:
            if param_name in self.updatable:
                param_len = self._get_internal_len(param_name)
                end_id = begin_id + param_len
                setattr(self, param_name, new_values[begin_id:end_id])
                begin_id = end_id
            else:
                msg = 'Attempted to set \'' + param_name + '\', but '
                msg += 'updatable parameters are ' + str(self.updatable)
                msg += '!'
                raise ValueError(msg)


class AdvancedStateOfHealth(GeneralContainer,
                            arbitrary_types_allowed=True,
                            validate_assignment=True):
    """
    Holds the collection of HealthParameters that defines the A-SOH.
    """
    def updatable_parameter_values(self) -> np.ndarray:
        asoh = []
        for health_param in self.names:
            asoh += getattr(self, health_param).get_updatable_parameter_values()
        return np.array(asoh)


class JointState(BaseModel,
                 arbitrary_types_allowed=True,
                 validate_assignment=True):
    """
    This class is used to hold the joint state of a model at a given instant.
    That is, it stores the physical transient hidden state (example: SOC, etc.),
    as well as the A-SOH model parameters (example: Q_total, R_0, etc.)
    """
    transient_state: HiddenVector = \
        Field(description='Physical transient hidden state')
    asoh: AdvancedStateOfHealth = \
        Field(description='Advanced State of Health (A-SOH)')

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.transient_state.names + self.asoh.names

    def to_numpy(self) -> np.ndarray:
        return np.hstack((self.transient_state.to_numpy(),
                          self.asoh.to_numpy()))


class HealthModel():
    """
    Base health model. At a minimum, it must be able to:
        1. given physical transient hidden state(s) and the A-SOH(s), output
            corresponding terminal voltage prediction(s)
        2. given physical transient hidden state(s), A-SOH(s), and new input(s),
            output new physical transient hidden state(s)
    """

    @abstractmethod
    def update_transient_state(
            self,
            transient_state: Union[HiddenVector, List[HiddenVector]],
            input: InputQuantities,
            asoh: Union[AdvancedStateOfHealth, List[AdvancedStateOfHealth]],
            *args, **kwargs) -> HiddenVector:
        pass

    @abstractmethod
    def predict_output(
            self,
            transient_state: Union[HiddenVector, List[HiddenVector]],
            input: InputQuantities,
            asoh: Union[AdvancedStateOfHealth, List[AdvancedStateOfHealth]],
            *args, **kwargs) -> OutputMeasurements:
        """
        Compute expected output (terminal voltage, etc.) of a the model.
        """
        pass
