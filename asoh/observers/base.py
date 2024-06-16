from asoh.models.base import ControlState, HealthModel, InstanceState, Outputs


class BaseEstimator:
    """Base class for all state estimation methods

    Using an Estimator
    ------------------

    Each instance of an Estimator tracks the estimates for states of single storage system.

    Start by supplying at least the model used to describe the dynamics of a system
    as a :class:`~asoh.models.base.HealthModel` and an initial guess for the
    state of the system `~asoh.models.base.InstanceState`.
    Some Estimators have additional parameters which define how they function. 

    .. note ::
     
        Recall that the state of a system includes the **state** parameters of the system
        (e.g., the state of charge) and parameters which define its dynamics (e.g., internal resistances).
        Some of the parameters which define the dynamics can be varied with time,
        and we denote those parameters **state of health**.
        
    Increase the states

    Args:
        model: Model which describes dynamics
        state: Initial guess for state and covariance matrix between state variables
    """

    model: HealthModel
    """Model which defines the dynamics of the system being estimated"""

    state: InstanceState
    """State estimate provided to the estate estimator"""

    def __init__(self, model: HealthModel, state: InstanceState):
        self.model = model
        self.state = state

    def step(self, u: ControlState, y: Outputs, t_step: float):
        """Update the state estimation given a new set of control states

        Args:
            u: Control states at the new time step
            y: Observed outputs at the new time step
            t_step: Time elapsed between last and current timestep
        """
        raise NotImplementedError()
