"""Base classes which define the state of a storage system,
the control signals applied to it, the outputs observable from it,
and the mathematical model which links state, control, and outputs together."""

from typing import Union, Optional, List
from abc import abstractmethod

import numpy as np
from pydantic import BaseModel, Field


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
                 additional_inputs: tuple[str, ...]=()) -> np.ndarray:
        """
        Outputs a numpy.ndarray where each entry is a np.ndarray corresponding 
        to one input vector. This input vector ALWAYS has as its first two 
        elements [time, current]. If provided, 'temperature' is the third 
        element. Other additional inputs are appear in the order provided
        """
        combined=[]
        for field in self.names:
            value=getattr(self, field)
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
                 additional_outputs: tuple[str, ...]=()) -> np.ndarray:
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
        transient_state = tuple(getattr(self, hid_var) \
                                for hid_var in self.names)
        return np.hstack(transient_state)


class HealthParameter(BaseModel,
                      arbitrary_types_allowed=True,
                      validate_assignment=True):
    """
    Base definition for a health parameter, such as Q_total, R_0, etc.
    """
    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        pass


class AdvancedStateOfHealth(GeneralContainer,
                            arbitrary_types_allowed=True,
                            validate_assignment=True):
    """
    Holds the collection of HealthParameters that defines the A-SOH 
    """
    def to_numpy(self,
                 additional_health: tuple[str, ...]=()) -> np.ndarray:
        asoh = np.array([])
        for health_param in self.names:
            asoh = np.hstack((asoh, getattr(self, health_param).to_numpy()))
        return asoh


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
        Compute expected terminal voltage of a the model
        """
        pass