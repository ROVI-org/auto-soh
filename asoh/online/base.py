"""Base class for online estimators and their related data structures"""

import numpy as np
from pydantic import BaseModel, Field

from asoh.models.base import InputState, HealthModel, SystemState, Measurements


class UpdateResult(BaseModel, extra='allow', arbitrary_types_allowed=True):
    """Result of updating the state using an observer"""

    y_err: np.ndarray = Field(..., description='Difference between estimated and observed outputs')


class OnlineEstimator:
    """Base class for state estimation methods which update incrementally.

    Using an OnlineEstimator
    ------------------------

    Each instance of an OnlineEstimator tracks the estimates for states of single storage system.

    Start by supplying at least the model used to describe the dynamics of a system
    as a :class:`~asoh.models.base.HealthModel` and an initial guess for the
    state of the system `~asoh.models.base.SystemState`.
    Some OnlineEstimators have additional parameters which define how they function.

    .. note ::

        Recall that the state of a system includes the **state** parameters of the system
        (e.g., the state of charge) and parameters which define its dynamics (e.g., internal resistances).
        Some of the parameters which define the dynamics can be varied with time,
        and we denote those parameters **state of health**.

    The :meth:`step` function of an OnlineEstimator adjusts the :attr:`state` given new observations.
    Invoke the step function given the observations at a new step,
    the control signal at the new time,
    and the time elapsed since the last step.
    The OnlineEstimator will use the update function of the supplied model to produce an
    estimate for the new state, then apply a correction logic to the observed state.
    The `step` function will return diagnostic signals relating to how the state
    was adjusted given new observations.

    Implementing an OnlineEstimator
    -------------------------------

    The :meth:`step` method is the only one which need be implemented for a new OnlineEstimator.
    Consider creating a subclass of :class:`UpdateResult` which defines definitions for
    any new diagnostic information produced by the new observer.

    Args:
        model: Model which describes dynamics
        state: Initial guess for state and covariance matrix between state variables.
            The estimator will create a copy of the input state.
    """

    model: HealthModel
    """Model which defines the dynamics of the system being estimated"""

    state: SystemState
    """State estimate provided to and updated by the observer"""

    def __init__(self, model: HealthModel, state: SystemState):
        self.model = model
        self.state = state.copy(deep=True)

    def step(self, u: InputState, y: Measurements, t_step: float) -> UpdateResult:
        """Update the state estimation given a new set of control states

        Args:
            u: Control states at the new time step
            y: Observed outputs at the new time step
            t_step: Time elapsed between last and current timestep
        Returns:
            Diagnostic information
        """
        raise NotImplementedError()
