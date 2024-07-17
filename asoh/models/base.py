"""Base classes which define the state of a storage system,
the control signals applied to it, the outputs observable from it,
and the mathematical model which links state, control, and outputs together."""
from typing import Iterator, Optional, List, Tuple, Dict, Union, Iterable, Sized
from abc import abstractmethod
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
    to a vector and update the values of the system health back into the class structure from a vector.

    ``HealthVariable`` will often be composed of submodels that are other ``HealthVariable`` or
    tuples and dictionaries of ``HealthVariable``.
    The following class shows a simple health model for a battery:

    .. code-block:: python

        class Resistance(HealthVariable):
            full: float
            '''Resistance at fully charged'''
            empty: float
            '''Resistance at fully discharged'''

            def get_resistance(self, soc: float):
                return self.empty + soc * (self.full - self.empty)

        class BatteryHealth(HealthVariable):
            capacity: float
            resistance: Resistance

        model = BatteryHealth(capacity=1., resistance={'full': 0.2, 'empty': 0.1})

    Accessing the Values of Parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Access the value of a parameter from the Python attributes

    .. code-block:: python

        assert model.resistance.full == 0.2

    or indirectly using :meth:`get_parameters`, which returns a list of floats

    The name of a variable within such hierarchical model contains the path to the submodel
    and the name of the attribute of the submodel separated by periods.
    For example, the resistance at fully charged of the following class is named "resistance.empty".

    .. code-block:: python

        assert model.get_parameters(['resistance.full']) == [0.2]


    Controlling which Parameters are Updatable
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    No parameters of the ``HealthVariable`` are treated as updatable by default.
    Mark a variable as updatable by marking the submodel(s) holding that variable as updatable and
    the variable as updatable in the submodel which holds by adding the names to the :attr:`updatable`
    set held by every ``HealthVariable`` class.
    Marking "resistance.empty" is achieved by

    .. code-block:: python

        model.updatable.add('resistance')
        model.resistance.updatable.add('empty')

    or using the :meth:`mark_updatable` utility method

    .. code-block:: python

        model.mark_updatable('resistance.empty')

    All submodels along the path to a specific parameter must be marked as updatable for that
    variable to be treated updatable. For example, "resistance.full" would not be considered updatable if
    the "resistance" submodel is not updatable

    .. code-block:: python

        model.updatable.remove('resistance')
        model.resistance.mark_updatable('full')  # Has no effect yet because 'resistance' is fixed

    Setting Values of Parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Parameters which are marked as updatable can be altered using :meth:`update_parameters`.

    Provide a list of new values and a list of names

    .. code-block:: python

        model.updatable.add('resistance')  # Allows resistance fields to be updated
        model.update_parameters([0.1], ['resistance.full'])

    or omit the specific names to set all updatable variables

    .. code-block:: python

        assert model.updatable_names == ['resistance.full', 'resistance.empty']
        model.update_parameters([0.2, 0.1])
    """

    updatable: set[str] = Field(default_factory=set)
    """Which fields are to be treated as updatable by a parameter estimator"""

    @property
    def num_updatable(self):
        """Number of updatable parameters in this HealthVariable"""
        return sum(len(x) for _, x in self.iter_parameters())

    @property
    def updatable_names(self) -> list[str]:
        """Names of all updatable parameters"""
        return list(k for k, _ in self.iter_parameters())

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

    def mark_updatable(self, name: str):
        """Mark a specific variable as updatable

        Will mark any submodel along the path to the requested name as updatable.

        Args:
            name: Name of the variable to be set as updatable
        """

        for n, m in zip(*self._get_model_chain(name)):
            m.updatable.add(n)

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

        # Get the model which holds this value
        names, models = self._get_model_chain(name)
        name, model = names[-1], models[-1]

        # Set appropriately
        setattr(model, name, value)

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

    def get_parameters(self, names: Optional[list[str]] = None) -> List[float]:
        """Get updatable parameters as a numpy vector

        Args:
            names: Names of the parameters to gather. If ``None``, then will return all updatable parameters
        Returns:
            A numpy array of the values
        """

        # Get all variables if no specific list is specified
        if names is None:
            names = list(k for k, v in self.iter_parameters())

        output = []
        for name in names:
            my_names, my_models = self._get_model_chain(name)
            value = getattr(my_models[-1], my_names[-1])
            if isinstance(value, Iterable):
                output.extend(value)
            else:
                output.append(value)
        return output

    def update_parameters(self, values: Union[np.ndarray, list[float]], names: Optional[list[str]] = None):
        """Set the value for updatable parameters given their names

        Args:
            values: Values of the parameters to set
            names: Names of the parameters to set. If ``None``, then will set all updatable parameters
        """

        # Get all variables if no specific list is specified
        if names is None:
            names = list(k for k, v in self.iter_parameters())

        end = pos = 0
        for name in names:
            # Get the associated model, and which attribute to set
            my_names, models = self._get_model_chain(name)

            # Raise an error if the attribute being set is not updatable
            for i, (n, m) in enumerate(zip(my_names, models)):
                if n not in m.updatable:
                    raise ValueError(
                        f'Variable {name} is not updatable because {n} '
                        f'is not updatable in self.{".".join(my_names[:i])}'
                    )

            # Get the number of parameters
            model = models[-1]
            attr = my_names[-1]
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


class GeneralContainer(BaseModel,
                       arbitrary_types_allowed=True):
    """
    General container class to store variables that are all numeric (that is, either floats or numpy arrays)
    """

    @property
    def all_fields(self) -> tuple[str, ...]:
        return tuple(self.model_fields.keys())

    def length_field(self, field_name: str) -> int:
        """
        Returns length of provided field name. If the field is a float, returns 1, otherwise, returns length of array.
        If field is None, returns 0.
        """
        field_val = getattr(self, field_name, None)
        if field_val is None:
            return 0
        elif isinstance(field_val, Sized):
            return len(field_val)
        return 1

    def to_numpy(self) -> np.ndarray:
        """
        Outputs everything that is stored as a np.ndarray
        """
        relevant_vals = tuple()
        for field_name in self.all_fields:
            field = getattr(self, field_name, None)
            if field is not None:
                relevant_vals += (field,)
        return np.hstack(relevant_vals)

    def from_numpy(self, values: np.ndarray) -> None:
        """
        Updates field values from a numpy array
        """
        # We need to know where to start reading from in the array
        begin_index = 0
        for field_name in self.all_fields:
            field_len = self.length_field(field_name)
            end_index = begin_index + field_len
            new_field_values = values[begin_index:end_index]
            setattr(self, field_name, new_field_values)
            begin_index = end_index


class InputQuantities(GeneralContainer):
    """The control of a battery system, such as the terminal current

    Add new fields to subclassess of ``ControlState`` for more complex systems
    """
    time: float = Field(description='Timestamp(s) of inputs. Units: s')
    current: float = Field(description='Current applied to the storage system. Units: A')


class OutputQuantities(GeneralContainer):
    """Output for observables from a battery system

    Add new fields to subclasses of ``ControlState`` for more complex systems
    """

    terminal_voltage: float = \
        Field(description='Voltage output of a battery cell/model. Units: V')


class TransientVector(GeneralContainer):
    """
    Stores physical transient/instantenous hidden state
    """
    pass


class AdvancedStateOfHealth(HealthVariable):
    """
    Stores A-SOH
    """
    pass


class CellModel():
    """
    Base cell model. At a minimum, it must be able to:
        1. given physical transient hidden state(s) and the A-SOH(s), output
            corresponding terminal voltage prediction(s)
        2. given a past physical transient hidden state(s), A-SOH(s), and new
            input(s), output new physical transient hidden state(s)
    """

    @abstractmethod
    def update_transient_state(
            self,
            input: InputQuantities,
            transient_state: TransientVector,
            asoh: AdvancedStateOfHealth,
            *args, **kwargs) -> TransientVector:
        pass

    @abstractmethod
    def calculate_terminal_voltage(
            self,
            input: InputQuantities,
            transient_state: TransientVector,
            asoh: AdvancedStateOfHealth,
            *args, **kwargs) -> OutputQuantities:
        """
        Compute expected output (terminal voltage, etc.) of the model.
        """
        pass
