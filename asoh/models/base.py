"""Base classes which define the state of a storage system,
the control signals applied to it, the outputs observable from it,
and the mathematical model which links state, control, and outputs together."""
from typing import Iterator, Optional, List, Tuple, Dict, Union
import logging

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# TODO (wardlt): Decide on what we call a parameter and a variable (or, rather, adopt @vventuri's terminology)
# TODO (wardlt): Make an "expand names" function to turn the name of a subvariable to a list of updatable names
# TODO (wardlt): Consider making a special "list of Variables" class provides the same `update_function`.
class HealthVariable(BaseModel, arbitrary_types_allowed=True):
    """Base class for a container which holds the physical parameters of system and which ones
    are being treated as updatable.

    Creating a System Health
    ------------------------

    Define a new system state by subclassing ``HealthVariable`` then providing
    adding attributes which describe the learnable parameters of a system.

    The attributes can be either singular or lists of floats,
    other ``HealthVariable`` classes,
    or lists or dictionaries of other ``HealthVariable`` classes.

    Using a System Health
    ---------------------

    The core purpose of the ``HealthVariable`` class is to serialize the parameters of system health
    to a numpy vector and update the values of the system health back into the class structure from a numpy vector.
    """

    updatable: set[str] = Field(default_factory=set)
    """Which fields are to be treated as updatable by a parameter estimator"""

    @property
    def num_parameters(self):
        """Number of updatable parameters in this class object"""
        return sum(len(x) for _, x in self.iter_parameters())

    def make_all_updatable(self, recurse: bool = True):
        """Make all fields in the model updatable

        Args:
            recurse: Make all parameters of each submodel updatable too
        """

        _allowed_field_types = (float, np.ndarray, HealthVariable, List, Tuple, Dict)
        for key in self.model_fields:
            # Add the field as updatable
            field = getattr(self, key)
            if isinstance(field, _allowed_field_types):
                self.updatable.add(key)

            # Recurse into submodels
            if not recurse:
                continue
            elif isinstance(field, HealthVariable):
                getattr(self, key).make_all_updatable()
            elif isinstance(field, (List, Tuple)):
                for submodel in getattr(self, key):
                    submodel.make_all_updatable()
            elif isinstance(field, Dict):
                for submodel in getattr(self, key).values():
                    submodel.make_all_updatable()

    def _get_associated_model(self, name: str) -> 'HealthVariable':
        """Get the object which stores the parameters associated with a certain variable name

        Used in the "get" and "update" operations to provide access to the location where
        the reference associated with a variable is held, which will allow us to update it

        Args:
            names: List of variables to acquire
        Yields:
            The instance of the model which holds each variable, in the order the names are provided
        """

        if '.' not in name:
            # Then we have reached the proper object
            return self
        else:
            # Determine which object to recurse into
            my_name, next_name = name.split(".", maxsplit=1)
            attr = getattr(self, my_name)

            if isinstance(attr, HealthVariable):
                return attr._get_associated_model(next_name)
            elif isinstance(attr, tuple):
                # Recurse into the right member off the list
                my_ind, next_name = next_name.split(".", maxsplit=1)
                my_attr: HealthVariable = attr[int(my_ind)]
                return my_attr._get_associated_model(next_name)
            elif isinstance(attr, dict):
                my_key, next_name = next_name.split(".", maxsplit=1)
                next_attr: HealthVariable = attr[my_key]
                return next_attr._get_associated_model(next_name)
            else:
                raise ValueError('There should be no other types of container')

    def set_value(self, name: str, value: Union[float, np.ndarray]):
        """Set the value of a certain variable by name

        Args:
            name: Name of the parameter to set.
            value: Updated value
        """

        if '.' not in name:
            attr = getattr(self, name)
            # TODO (wardlt); Allow setting all variables of a HealthVariable in one go
            if not isinstance(attr, (float, np.ndarray)):
                raise ValueError(f'{name} is a health variable or collection of health variables.'
                                 ' You must provide the name of which attribute in that variable to set.')

            # The value belongs to this object
            setattr(self, name, value)
        else:
            my_name, next_name = name.split(".", maxsplit=1)
            attr = getattr(self, my_name)

            if isinstance(attr, HealthVariable):
                attr.set_value(next_name, value)
            elif isinstance(attr, tuple):
                my_ind, next_name = next_name.split(".", maxsplit=1)
                my_attr = attr[int(my_ind)]
                my_attr.set_value(next_name, value)
            elif isinstance(attr, dict):
                my_key, next_name = next_name.split(".", maxsplit=1)
                attr[my_key].set_value(next_name, value)
            else:
                raise ValueError('There should be no other types of container')

    # TODO (wardlt): Will we ever need to iterate over all parameters, not just the updatable ones
    # TODO (wardlt): Document that if a field is marked as "fixed" in the top class, any annotation of it as
    #  updatable in the subclasses will be ignored
    def iter_parameters(self, updatable_only: bool = True, recurse: bool = True) -> Iterator[tuple[str, np.ndarray]]:
        """Iterate over all parameters which are treated as updatable

        Args:
            updatable_only: Only iterate over variables which are updatable
            recurse: Whether to gather parameters from attributes which are also ``HealthVariable`` classes.
        Yields:
            Tuple of names and parameter values as numpy arrays. The name of parameters from attributes
            which are ``HealthVariable`` will start will be
            "<name of attribute in this class>.<name of attribute in submodel>"
        """

        for key in self.model_fields:  # Iterate over fields and not updatable to have repeatable order
            if updatable_only and key not in self.updatable:
                continue

            field = getattr(self, key)
            if isinstance(field, float):  # TODO (wardlt): Will we ever treat integer fields as updatable?
                yield key, np.array([getattr(self, key)])
            elif isinstance(field, np.ndarray):
                yield key, getattr(self, key)
            elif isinstance(field, HealthVariable) and recurse:
                submodel: HealthVariable = getattr(self, key)
                for subkey, subvalue in submodel.iter_parameters(updatable_only=updatable_only, recurse=recurse):
                    yield f'{key}.{subkey}', subvalue
            elif isinstance(field, (List, Tuple)) and recurse:
                submodels: List[HealthVariable] = getattr(self, key)
                for i, submodel in enumerate(submodels):
                    for subkey, subvalue in submodel.iter_parameters(updatable_only=updatable_only, recurse=recurse):
                        yield f'{key}.{i}.{subkey}', subvalue
            elif isinstance(field, Dict) and recurse:
                submodels: Dict[str, HealthVariable] = getattr(self, key)
                for subkey, submodel in submodels.items():
                    for subsubkey, subvalue in submodel.iter_parameters(updatable_only=updatable_only, recurse=recurse):
                        yield f'{key}.{subkey}.{subsubkey}', subvalue
            else:
                logger.debug(f'The "{key}" field is not any of the type associated with health variables, skipping')

    def get_parameters(self, names: Optional[set[str]] = None) -> np.ndarray:
        """Get updatable parameters as a numpy vector

        Args:
            names: Names of the parameters to gather. If ``None``, then will return all updatable parameters
        Returns:
            A numpy array of the values
        """
        raise NotImplementedError()

    def update_parameters(self, values: np.ndarray, names: Optional[list[str]] = None):
        """Set the value for updatable parameters given their names

        Args:
            values: Values of the parameters to set
            names: Names of the parameters to set. If ``None``, then will return all updatable parameters
        """

        # Get all variables if no specific list is specified
        if names is None:
            names = list(k for k, v in self.iter_parameters())

        end = pos = 0
        for name in names:
            # Get the associated model, and which attribute to set
            model = self._get_associated_model(name)
            attr = name.rsplit(".", maxsplit=1)[-1] if '.' in name else name

            # Get the number of parameters
            cur_value = getattr(model, attr)
            num_params = np.size(cur_value)

            # Get the parameters to use
            end = pos + num_params
            if pos + num_params > len(values):
                raise ValueError(f'Required at least {end} values, but only provided {len(values)}')
            new_value = values[pos:end]
            setattr(model, attr, new_value)

            # Increment the starting point for the next parameter
            pos = end

        # Check to make sure all were used
        if end != len(values):
            raise ValueError(f'Did not use all parameters. Provided {len(values)}, used {end}')


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


class CellModel():
    """
    Base health model. At a minimum, it must be able to:
        1. given physical transient hidden state(s) and the A-SOH(s), output
            corresponding terminal voltage prediction(s)
        2. given a past physical transient hidden state(s), A-SOH(s), and new 
            input(s), output new physical transient hidden state(s)
    """

    @abstractmethod
    def update_transient_state(
            self,
            input: InputQuantities,
            transient_state: Union[HiddenVector, List[HiddenVector]],
            asoh: Union[AdvancedStateOfHealth, List[AdvancedStateOfHealth]],
            *args, **kwargs) -> HiddenVector:
        pass

    @abstractmethod
    def get_voltage(
            self,
            input: InputQuantities,
            transient_state: Union[HiddenVector, List[HiddenVector]],
            asoh: Union[AdvancedStateOfHealth, List[AdvancedStateOfHealth]],
            *args, **kwargs) -> OutputMeasurements:
        """
        Compute expected output (terminal voltage, etc.) of a the model.
        """
        pass
