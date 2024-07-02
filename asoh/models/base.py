"""Base classes which define the state of a storage system,
the control signals applied to it, the outputs observable from it,
and the mathematical model which links state, control, and outputs together."""
from typing import Iterator, Optional

import numpy as np
from pydantic import BaseModel, Field


class ParameterSet(BaseModel, arbitrary_types_allowed=True):
    """Base class for a container which holds the physical parameters of system and which ones
    are being treated as updatable.

    Creating a System Health
    ------------------------

    Define a new system state by subclassing ``ParameterSet`` then providing
    adding attributes which describe the learnable parameters of a system.

    The attributes can be either singular or lists of floats,
    other ``ParameterSet`` classes,
    or lists or dictionaries of other ``ParameterSet`` classes.

    Using a System Health
    ---------------------

    The core purpose of the ``ParameterSet`` class is to serialize the parameters of system health
    to a numpy vector and update the values of the system health back into the class structure from a numpy vector.

    """

    updatable: set[str] = Field(default_factory=set)
    """Which fields are to be treated as updatable by a parameter estimator"""

    @property
    def num_parameters(self):
        """Number of updatable parameters in this class object"""
        return sum(len(x) for _, x in self.iter_parameters())

    def iter_parameters(self, recurse: bool = True) -> Iterator[tuple[str, np.ndarray]]:
        """Iterate over all parameters which are treated as updatable

        Args:
            recurse: Whether to gather parameters from attributes which are also ``ParameterSet`` classes.
        Yields:
            Tuple of names and parameter values as numpy arrays. The name of parameters from attributes
            which are ``ParameterSet`` will start will be
            "<name of attribute in this class>.<name of attribute in submodel>"
        """
        raise NotImplementedError()

    def count_parameters(self, names: Optional[set[str]] = None) -> int:
        """Count the number of parameters for this model

        Args:
            names: Names of the parameters to count. If ``None``, then will count all updatable parameters
        """
        raise NotImplementedError()

    def get_parameters(self, names: Optional[set[str]] = None) -> np.ndarray:
        """Get updatable parameters as a numpy vector

        Args:
            names: Names of the parameters to gather. If ``None``, then will return all updatable parameters
        Returns:
            A numpy array of the values
        """
        raise NotImplementedError()

    def set_parameters(self, values: np.ndarray, names: Optional[set[str]] = None):
        """Set the value for learnable parameters given their names

        Args:
            values: Values of the parameters to set
            names: Names of the parameters to set. If ``None``, then will return all updatable parameters
        """
        raise NotImplementedError()


class InputState(BaseModel):
    """The control of a battery system, such as the terminal current

    Add new fields to subclassess of ``ControlState`` for more complex systems
    """

    current: float = Field(description='Current applied to the storage system. Units: A')

    def to_numpy(self) -> np.ndarray:
        """Control inputs as a numpy vector"""
        output = [getattr(self, key) for key in self.model_fields.keys()]
        return np.array(output)


class Measurements(BaseModel):
    """Output for observables from a battery system

    Add new fields to subclasses of ``ControlState`` for more complex systems
    """

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.model_fields.keys())

    def to_numpy(self) -> np.ndarray:
        """Outputs as a numpy vector"""
        output = [getattr(self, key) for key in self.model_fields.keys()]
        return np.array(output)
