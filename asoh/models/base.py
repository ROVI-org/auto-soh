"""Base classes which define the state of a storage system,
the control signals applied to it, the outputs observable from it,
and the mathematical model which links state, control, and outputs together."""

import numpy as np
from pydantic import BaseModel, Field


class InstanceState(BaseModel):
    """Defines the state of a particular instance of a model

    Creating a InstanceState
    ------------------------

    Define a new instance state by adding the attributes which define the state
    of a particular model to a subclass of ``InstanceState``. List the names of
    attributes which always vary with time as the :attr:`state_params`.
    """

    health_params: tuple[str, ...] = Field(description='List of parameters which are being treated as state of health'
                                                       ' for this particular instance of a storage system', default_factory=tuple)

    state_params: tuple[str, ...] = ...
    """Parameter which are always treated as time dependent"""

    def _assemble_array(self, params: tuple[str, ...]) -> np.ndarray:
        """Assemble a numpy array from the instances within this class

        Args:
            params: Names of parameters to store
        Returns:
            A numpy array of the specified parameters
        """
        output = []
        for s in params:
            x = getattr(self, s)
            if isinstance(x, (float, int)):
                output.append(x)
            else:
                output.extend(x)
        return np.array(output)

    @property
    def state(self) -> np.ndarray:
        """Only the state of variables"""
        return self._assemble_array(self.state_params)

    @property
    def soh(self) -> np.ndarray:
        """Only the state of health variables"""
        return self._assemble_array(self.health_params)

    @property
    def full_state(self) -> np.ndarray:
        return self._assemble_array(self.state_params + self.health_params)


class ControlState(BaseModel):
    """The control of a battery system

    Add new fields to subclassess of ``ControlState`` for more complex systems
    """

    current: float = Field(description='Current applied to the storage system. Units: A')


class OutputModel(BaseModel):
    """Output model for observables from a battery system

    Add new fields to subclasses of ``ControlState`` for more complex systems
    """


class HealthModel:
    """A model for a storage system which describes its operating state and health.

    Types of Parameters
    -------------------

    A storage system is described by many parameters differentiated by whether they remain static
    or change during operation.

    Static parameters define the design elements of a system that are not expected to degrade.

    Dynamic parameters could be those which change rapidly with operating conditions,
    like the state of charge, or those which change slowly with time, like the internal resistances.
    The quickly-varying parameters are called the **state** of the battery.
    The slowly-varying parameters are called the **state-of-health** of the battery.

    Implementing a Health Model
    ---------------------------

    First create the states which define your model as subclasses of the
    :class:`InstanceState`, :class:`ControlState`, and :class:`OutputModel`.

    We assume all models express dynamic systems which vary continuously with time.
    Implement the derivatives of each model parameter as a function of time
    as the :meth:`dx` function.

    Define the output function as :meth:`output`.
    """

    def dx(self, state: InstanceState, control: ControlState) -> np.ndarray:
        """Compute the derivatives of each state variable with respect to time

        Args:
            state: State of the battery system
            control: Control signal applied to the system

        Returns:
            The derivative of each parameter with respect to time in units of "X per second"
        """
        raise NotImplementedError()

    def output(self, state: InstanceState, control: ControlState) -> OutputModel:
        """Compute the observed outputs of a system given the current state and control

        Args:
            state: State of the battery system
            control: Control signal applied to the system

        Returns:
            Each observable of the battery system
        """
        raise NotImplementedError()
