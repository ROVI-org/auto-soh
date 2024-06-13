"""Base classes for objects which represent energy storage systems"""
from collections.abc import Collection

from progpy.prognostics_model import PrognosticsModel
from progpy.utils.containers import StateContainer


class HealthModel(PrognosticsModel):
    """A prognostic model which allows toggling parameters between fixed descriptions of system
     and dynamic variables which evolve with time.

    Implementing a Health Model
    ---------------------------

    A typical ProgPy :class:`~progpy.PrognosticsModel` requires expressing a difference between
    "states," which evolve with time, and "parameters," which remain constant.
    The `states` attribute of the prognostic model lists the names of the states
    and the names and default parameters are all listed in the `default_parameters` dictionary,
    with the "states" supplied in the "x0" key of `default_parameters`

    The parameters used as states for a `HealthModel` fall into two categories,
    those which are always state variables (e.g., the state of charge of a battery)
    and those which are allowed to change with time (e.g., the capacity of a battery).
    As such, the `default_parameters` and `states` of a model must change based on
    which parameters the user chooses as adjustable.

    STOPPED HERE WHILE I MADE SURE THIS WORKED

    """

    _always_states: dict[str, float] = ...
    """Names and default values of that are always state variables"""

    _health_parameters: dict[str, float] = ...
    """Names and default values of health parameters, which may or may not be treated as adjustable"""

    health_states = tuple[str]
    """Names of parameters which describe the state of health of a system"""

    @property
    def default_parameters(self) -> dict[str, float | dict[str, float]]:
        output = {'x0': self._always_states.copy()}
        for key, val in self._health_parameters.items():
            if key in self.health_states:
                output['x0'][key] = val
            else:
                output[key] = val
        return output

    @property
    def states(self) -> list[str]:
        return list(self._always_states.keys()) + list(self.health_states)

    def __init__(self, health_states: Collection[str], **kwargs):
        self.health_states = tuple(health_states)
        super().__init__(**kwargs)

    def _combine_states_and_parameters(self, x: StateContainer) -> dict[str, float]:
        """Return a dictionary where any "health" parameters overwrite the default values used for the model

        Args:
            x: Input state container
        Returns:
            Un-nested dictionary containing both the states, adjustable parameters, and fixed parameters
        """

        output = self._health_parameters.copy()
        output.update(x)
        return output

    def dx(self, x, u):
        # The change in health parameters is always zero wrt time
        return dict((k, 0.) for k in self.health_states)
