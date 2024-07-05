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

    def mark_all_updatable(self, recurse: bool = True):
        """Make all fields in the model updatable

        Args:
            recurse: Make all parameters of each submodel updatable too
        """

        _allowed_field_types = (float, np.ndarray, HealthVariable, List, Tuple, Dict)

        models = self._iter_over_submodels() if recurse else (self,)
        for model in models:
            for key in model.model_fields.keys():
                # Add the field as updatable
                field = getattr(model, key)
                if isinstance(field, _allowed_field_types):
                    model.updatable.add(key)

    def mark_all_fixed(self, recurse: bool = True):
        """Mark all fields in the model as not updatable

        Args:
            recurse: Whether to mark all variables of submodels as not updatable
        """

        models = self._iter_over_submodels() if recurse else (self,)
        for model in models:
            model.updatable.clear()

    def _iter_over_submodels(self) -> Iterator['HealthVariable']:
        """Iterate over all models which compose this HealthVariable"""

        yield self
        for key in self.model_fields:
            field = getattr(self, key)
            if isinstance(field, HealthVariable):
                yield from field._iter_over_submodels()
            elif isinstance(field, (List, Tuple)):
                for submodel in getattr(self, key):
                    yield from submodel._iter_over_submodels()
            elif isinstance(field, Dict):
                for submodel in getattr(self, key).values():
                    yield from submodel._iter_over_submodels()

    def _get_model_chain(self, name: str) -> tuple[tuple[str, ...], tuple['HealthVariable', ...]]:
        """Get the series of ``HealthVariable`` associated with a certain parameter
        such that self is the first member of the tuple and the ``HealthVariable``
        which holds a reference to the value being request is the last.

        For example, the chain for attribute "a" is ``(self,)`` because the variable "a" belongs to self.
        The chain for attribute "b.a" is ``(self, self.b)`` because the variable "b.a" is attribute "a"
        of the HealthVariable which is attribute "b" of self.

        Used in the "get" and "update" operations to provide access to the location where
        the reference associated with a variable is held, which will allow us to update it

        Args:
            name: Name of the variable to acquire
        Returns:
            - The name of the attribute associated with the requested variable in each model along the chain
            - The chain of HealthVariable instances which holds each variable
        """

        if '.' not in name:
            # Then we have reached the proper object
            return (name,), (self,)
        else:
            # Determine which object to recurse into
            my_name, next_name = name.split(".", maxsplit=1)
            next_inst: HealthVariable = getattr(self, my_name)

            # Select the appropriate next model from the collection
            if isinstance(next_inst, HealthVariable):
                pass
            elif isinstance(next_inst, tuple):
                # Recurse into the right member off the list
                my_ind, next_name = next_name.split(".", maxsplit=1)
                next_inst = next_inst[int(my_ind)]
            elif isinstance(next_inst, dict):
                my_key, next_name = next_name.split(".", maxsplit=1)
                next_inst = next_inst[my_key]
            else:
                raise ValueError('There should be no other types of container')

            # Recurse
            next_names, next_models = next_inst._get_model_chain(next_name)
            return (my_name,) + next_names, (self,) + next_models

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
        # Get all variables if no specific list is specified
        if names is None:
            names = list(k for k, v in self.iter_parameters())

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
            names, models = self._get_model_chain(name)

            # Raise an error if the attribute being set is not updatable
            for i, (n, m) in enumerate(zip(names, models)):
                if n not in m.updatable:
                    raise ValueError(
                        f'Variable {name} is not updatable because {n} is not updatable in self.{".".join(names[:i])}'
                    )

            # Get the number of parameters
            model = models[-1]
            attr = names[-1]
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
