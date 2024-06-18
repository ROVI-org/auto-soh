"""Base classes which define the state of a storage system,
the control signals applied to it, the outputs observable from it,
and the mathematical model which links state, control, and outputs together."""

import numpy as np
from scipy.integrate import solve_ivp, OdeSolution
from pydantic import BaseModel, Field, model_validator


# TODO (wardlt): Consider an implementation where we store the parameters as a single numpy array, then provide helper classes for the parameters.
#  Maybe add some kind of `compile` which generates an object which provides such an interface (progpy does something like that),
#  which would [maybe] conflict with the desire to combine this class and HealthModel
class InstanceState(BaseModel, arbitrary_types_allowed=True):
    """Defines the state of a particular instance of a model

    Using an Instance State
    -----------------------

    Create a InstanceState by calling the constructor with the values of all parameters,
    and the lists of those parameters which are treated as the state of health.

    .. code-block: python

        state = InstanceState(charge=1., health_params=('capacity',))

    Access or set the state variables in an object-oriented style

    .. code-block: python

        assert state.attribute == 1
        state.attribute = 2

    or an array style:

    .. code-block: python:

        assert np.isclose(state.state, [2.]).all()
        state.set_state([3.])

    The array style can access or set three subsets of the parameters:

    1. :attr:`.state` and :meth:`set_state` for variables which are always transient
    2. :attr:`soh` and :meth:`set_soh` for variables which define the state of health of the system
    2. :attr:`full_state` and :meth:`set_full_state` for both sets of variables.

    Implementing a New State Class
    ------------------------------

    Define a new instance state by adding the attributes which define the state
    of a particular model to a subclass of ``InstanceState``. List the names of
    attributes which always vary with time as the :attr:`state_params`.

    The type of attributes can be either float or a tuple of floats.
    """

    health_params: tuple[str, ...] = Field(description='List of parameters which are being treated as state of health'
                                                       ' for this particular instance of a storage system', default_factory=tuple)

    covariance: np.ndarray = Field(None, description='Covariance matrix between all parameters being fit, including "health" and "state"')

    state_params: tuple[str, ...] = ...
    """Parameters which are always treated as time dependent"""

    @model_validator(mode='after')
    def _check_covariance(self):
        """Make it unit normal distribution to start with"""
        n_params = len(self.full_params)
        if self.covariance is None:
            self.covariance = np.eye(n_params)
        if self.covariance.shape != (n_params, n_params):
            raise ValueError(f'Expected ({n_params}, {n_params}) covariance matrix. Found {self.covariance.shape}')

    def _compile_names(self, params: tuple[str, ...]):
        """Expand parameter names where each member of a tuple parameter gets a unique name

        Args:
            params: Which parameters to expand
        Returns:
        """

        names = []
        for s in params:
            x = getattr(self, s)
            if isinstance(x, (float, int)):
                names.append(s)
            else:
                names.extend(f'{s}_{i}' for i in range(len(x)))
        return tuple(names)

    @property
    def state_names(self):
        """Names of the variables used as state"""
        return self._compile_names(self.state_params)

    @property
    def soh_names(self):
        """Names of the variables used as state of health"""
        return self._compile_names(self.health_params)

    @property
    def full_state_names(self):
        """Names of the variables used as state"""
        return self.state_names + self.soh_names

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
    def full_params(self) -> tuple[str, ...]:
        """All parameters being adjusted by the model"""
        return self.state_params + self.health_params

    @property
    def state(self) -> np.ndarray:
        """Values of the state variables"""
        return self._assemble_array(self.state_params)

    @property
    def soh(self) -> np.ndarray:
        """Values of the state of health variables"""
        return self._assemble_array(self.health_params)

    @property
    def full_state(self) -> np.ndarray:
        """Values of both the state of health and state variables"""
        return self._assemble_array(self.state_params + self.health_params)

    @property
    def num_params(self) -> int:
        """Number of parameters being fit, including both state and health parameters"""
        return len(self.state_params) + len(self.state_params)

    def _update_params(self, x, params):
        """Update the parameters of parts of the state given the list of names and their new values"""
        param_iter = iter(x)
        for s in params:
            x = getattr(self, s)
            if isinstance(x, (float, int)):
                setattr(self, s, next(param_iter))
            else:
                p = [next(param_iter) for _ in x]
                setattr(self, s, p)

    def set_state(self, new_state: np.ndarray | list[float]):
        """Update this state to the values in the vector

        Args:
            new_state: New parameters
        """
        self._update_params(new_state, self.state_params)

    def set_soh(self, new_state: np.ndarray | list[float]):
        """Update the state of health values given a list of values

        Args:
            new_state: New parameters
        """
        self._update_params(new_state, self.health_params)

    def set_full_state(self, new_state: np.ndarray | list[float]):
        """Update the state and health parameters given a list of values

        Args:
            new_state: New parameters
        """
        self._update_params(new_state, self.full_params)


class ControlState(BaseModel):
    """The control of a battery system

    Add new fields to subclassess of ``ControlState`` for more complex systems
    """

    current: float = Field(description='Current applied to the storage system. Units: A')

    def to_numpy(self) -> np.ndarray:
        """Control inputs as a numpy vector"""
        output = [getattr(self, key) for key in self.model_fields.keys()]
        return np.array(output)


class Outputs(BaseModel):
    """Output model for observables from a battery system

    Add new fields to subclasses of ``ControlState`` for more complex systems
    """

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.model_fields.keys())

    def to_numpy(self) -> np.ndarray:
        """Outputs as a numpy vector"""
        output = [getattr(self, key) for key in self.model_fields.keys()]
        return np.array(output)


# TODO (wardlt): Use generic classes? That might cement the relationship between a model and its associated input types
# TODO (warldt): Can we combine State and HealthModel? The State could contain parameters about how to perform updates
# TODO (warldt): Consider how to denote vectorized versions of each function
class HealthModel:
    """A model for a storage system which describes its operating state and health.

    Using a Health Model
    --------------------

    The health model implements tools which simulate using a storage system under specific control systems.
    Either call the :meth:`dx` function to estimate the rate of change of state parameters over time,
    or call :meth:`update` to simulate the change in state over a certain time period.

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

    Define the output function as :meth:`output` and how many outputs to expect as :attr:`num_outputs`
    """

    num_outputs: int = ...
    """Number of outputs from the output function"""

    def dx(self, state: InstanceState, control: ControlState) -> np.ndarray:
        """Compute the derivatives of each state variable with respect to time

        Args:
            state: State of the battery system
            control: Control signal applied to the system

        Returns:
            The derivative of each parameter with respect to time in units of "X per second"
        """
        raise NotImplementedError()

    def update(self, state: InstanceState, control: ControlState, total_time: float) -> OdeSolution:
        """Update the state under the influence of a control variable for a certain amount of time

        Args:
            state: Starting state
            control: Control signal
            total_time: Amount of time to propagate
        Returns:
            Result from SciPy's ODE solver
        """

        # Set up the time propagation function
        def _update_fun(t, y):
            # TODO (wardlt): Should we have control signals which vary with time? E.g., a linear interpolation between current/new state?
            state.set_state(y)
            return self.dx(state, control)

        result = solve_ivp(
            fun=_update_fun,
            y0=state.state,
            t_span=(0, total_time)
        )
        state.set_state(result.y[:, -1])
        return result

    def output(self, state: InstanceState, control: ControlState) -> Outputs:
        """Compute the observed outputs of a system given the current state and control

        Args:
            state: State of the battery system
            control: Control signal applied to the system

        Returns:
            Each observable of the battery system
        """
        raise NotImplementedError()
