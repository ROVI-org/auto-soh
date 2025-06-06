"""Base classes which define the state of a storage system,
the control signals applied to it, the outputs observable from it,
and the mathematical models which links state, control, and outputs together."""
from functools import cached_property
from typing import (
    Iterator, Optional, List, Tuple, Dict, Union, Any, Iterable, Sequence,
    Annotated, get_origin, get_args, Literal
)
from typing_extensions import Self
from abc import abstractmethod
import logging

import numpy as np
from pydantic import BaseModel, Field, BeforeValidator, model_validator, WrapSerializer

logger = logging.getLogger(__name__)


# Definitions for variables that should be single-valued and multi-valued
def enforce_dimensions(x: Any, dim=1) -> np.ndarray:
    """
    Make sure an array is the desired shape for batching

    Arrays must be 2D or greater and the first dimension is always the batch dimension.
    That means arrays which represent "scalar values" (dim == 0), have shape (batches, 1).

    Args:
        x: Value to be altered
        dim: Dimensionality of numbers being represented
    Returns:
        Array ready for use in a HealthVariable, etc
    """

    x = np.array(x)
    if dim == 0:
        x = np.squeeze(x)
        if x.ndim > 1:
            raise ValueError(f'Inconsistent dimensionality. Trying to store a {x.ndim} array as a scalar')
        return np.atleast_1d(x)[:, None]
    elif dim == 1:
        return np.atleast_2d(x)
    else:
        raise ValueError(f'We do not yet support arrays with dimensionality of {dim}')


def _encode_ndarray(value: np.ndarray, handler) -> List:
    """Encode a numpy array as a regular list"""
    return value.tolist()


ScalarParameter = Annotated[
    np.ndarray,
    Field(json_schema_extra={'type': 'moirae_parameter'}),
    BeforeValidator(lambda x: enforce_dimensions(x, 0)), Field(validate_default=True),
    WrapSerializer(_encode_ndarray, when_used='json-unless-none')
]
"""Type annotation for parameters which are exactly one value"""

ListParameter = Annotated[
    np.ndarray,
    Field(json_schema_extra={'type': 'moirae_parameter'}),
    BeforeValidator(lambda x: enforce_dimensions(x, 1)), Field(validate_default=True),
    WrapSerializer(_encode_ndarray, when_used='json-unless-none')
]
"""Type annotation for parameters which can be any number of values"""

NumpyType = Annotated[
    np.ndarray, BeforeValidator(np.array), Field(), WrapSerializer(_encode_ndarray, when_used='json-unless-none')
]
"""Type annotation for a field which is a numpy array but does not requires the batch dimensions used by parameters"""


# TODO (wardlt): Decide on what we call a parameter and a variable (or, rather, adopt @vventuri's terminology)
class HealthVariable(BaseModel, arbitrary_types_allowed=True):
    """Base class for a container which holds the physical parameters of system and which ones
    are being treated as updatable."""

    updatable: set[str] = Field(default_factory=set)
    """Which fields are to be treated as updatable by a parameter estimator"""

    @cached_property
    def all_fields_with_types(self) -> tuple[tuple[str, Literal['parameter', 'variable', 'sequence', 'dict']], ...]:
        """Names of all fields which correspond to physical parameters and their types

        Types of fields include:
            - `parameter`: a NumPy array of health parameter values
            - `variable`: another HealthVariable class
            - `sequence`: a list of HealthVariables
            - `dict`: a map of HealthVariables
        """
        # Filter out to only those which are either a Parameter, Health Variable or collection of Health Variables
        output = []
        for field, info in self.__class__.model_fields.items():
            # Simple case: it's a parameter
            if info.annotation == np.ndarray and info.json_schema_extra.get('type') == 'moirae_parameter':
                output.append((field, 'parameter'))
            elif get_origin(info.annotation) is None and issubclass(info.annotation, HealthVariable):
                output.append((field, 'variable'))
            elif get_origin(info.annotation) in (list, tuple) and \
                    issubclass(get_args(info.annotation)[0], HealthVariable):
                output.append((field, 'sequence'))
            elif get_origin(info.annotation) in (dict,) and \
                    issubclass(get_args(info.annotation)[1], HealthVariable):
                output.append((field, 'dict'))
        return tuple(output)

    @property
    def all_fields(self) -> tuple[str, ...]:
        """Names of all fields which correspond to physical parameters and their types"""
        return tuple(x for x, _ in self.all_fields_with_types)

    @model_validator(mode='after')
    def check_batch_size(self):
        assert self.batch_size > 0
        return self

    @property
    def batch_size(self) -> int:
        """Batch size of this parameter"""
        batch_size = 1
        batch_param_name = None  # Name of the parameter which is setting the batch size
        for name, param in self.iter_parameters(updatable_only=False, recurse=True):
            my_batch = param.shape[0]
            if batch_size > 1 and my_batch != 1 and my_batch != batch_size:
                raise ValueError(f'Inconsistent batch sizes. {name} has batch dim of {my_batch},'
                                 f' whereas {batch_param_name} has a size of {batch_size}')
            batch_size = max(batch_size, my_batch)
            if my_batch == batch_size:
                batch_param_name = name
        return batch_size

    @property
    def num_updatable(self):
        """Number of updatable parameters in this HealthVariable"""
        return sum(x.shape[-1] for _, x in self.iter_parameters())

    @property
    def updatable_names(self) -> Tuple[str, ...]:
        """Names of all updatable parameters"""
        return tuple(k for k, _ in self.iter_parameters())

    @property
    def all_names(self) -> Tuple[str, ...]:
        """Names of all updatable parameters"""
        return tuple(k for k, _ in self.iter_parameters(updatable_only=False))

    def expand_names(self, names: Iterable[str]) -> Tuple[str, ...]:
        """Expand names which define a collection of values to one for each number.

        Each member of a list of values become are annotated with ``[i]`` notation.

        .. code-block:: python

            class ListHealth(HealthVariable):
                x: ListParameter = 1.

            a = ListHealth()
            a.expand_names(['x'])  # == ['x[0]']

        Names of values that are themselves :class:`HealthVariable` are expanded to
        include all values

        .. code-block:: Python

            class Health(HealthVariable):
                a: ListHealth

            h = Health(a=a)
            h.expand_names(["a"])  # == ['a.x[0]']
            h.expand_names(["a.x"])  # == ['a.x[0]']

        Args:
            names: List of names to be expanded
        Returns:
            Expanded names
        """

        output = []
        for name in names:
            param = getattr(self, name.split(".", maxsplit=1)[0])

            # It is a value of this class
            if isinstance(param, np.ndarray):
                if param.shape[1] == 1:
                    output.append(name)  # It is a scalar
                else:
                    output.extend(f'{name}[{i}]' for i in range(param.shape[1]))
                continue

            # It is multiple values
            if isinstance(param, HealthVariable) and '.' not in name:  # Entire class
                sub_names = param.expand_names(param.all_names)
                prefix = name
            elif isinstance(param, HealthVariable):  # Specific field within class
                prefix, sub_name = name.split(".", maxsplit=1)
                sub_names = param.expand_names([sub_name])
            elif isinstance(param, (tuple, dict)):
                sub_count = name.count('.')
                is_tuple = isinstance(param, tuple)
                if sub_count == 0:  # All entry, all names
                    prefix = name
                    sub_names = []
                    for i, sub_param in enumerate(param) if is_tuple else param.items():
                        sub_names.extend(f'{i}.{n}' for n in sub_param.expand_names(sub_param.all_names))
                elif sub_count == 1:  # Single entry, all names
                    prefix, index = name.split(".")
                    sub_param = param[int(index) if is_tuple else index]
                    sub_names = [f'{index}.{n}' for n in sub_param.expand_names(sub_param.all_names)]
                else:  # Single entry, specific name
                    prefix, index, sub_name = name.split(".", maxsplit=2)
                    sub_param = param[int(index) if is_tuple else index]
                    sub_names = [f'{index}.{n}' for n in sub_param.expand_names([sub_name])]
            else:
                raise NotImplementedError('Unsupported type of nesting')

            output.extend(f'{prefix}.{n}' for n in sub_names)

        return tuple(output)

    def mark_all_updatable(self, recurse: bool = True):
        """Make all fields in the model updatable

        Args:
            recurse: Make all parameters of each submodel updatable too
        """
        models = self._iter_over_submodels() if recurse else (self,)
        for model in models:
            model.updatable.update(model.all_fields)

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
            if n not in m.all_fields:
                raise ValueError(f'Failed to mark {name}. '
                                 f'No such parameter {n} in health variable {m.__class__.__name__}')
            m.updatable.add(n)

    def _iter_over_submodels(self) -> Iterator['HealthVariable']:
        """Iterate over all models which compose this HealthVariable"""

        yield self
        for key, my_type in self.all_fields_with_types:
            field = getattr(self, key)
            if my_type == 'variable':
                yield from field._iter_over_submodels()
            elif my_type == 'sequence':
                for submodel in field:
                    yield from submodel._iter_over_submodels()
            elif my_type == 'dict':
                for submodel in field.values():
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

        # Turn into a numpy array
        cur_value: np.ndarray = getattr(model, name)
        dim = 0 if cur_value.shape[-1] == 1 else 1
        value = enforce_dimensions(value, dim)

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

        for key, info in self.all_fields_with_types:
            if updatable_only and key not in self.updatable:
                continue

            field = getattr(self, key)
            if info == 'parameter':
                yield key, field
            elif recurse and info == 'variable':
                submodel: HealthVariable = getattr(self, key)
                for subkey, subvalue in submodel.iter_parameters(updatable_only=updatable_only, recurse=recurse):
                    yield f'{key}.{subkey}', subvalue
            elif recurse and info == 'sequence':
                submodels: List[HealthVariable] = getattr(self, key)
                for i, submodel in enumerate(submodels):
                    for subkey, subvalue in submodel.iter_parameters(updatable_only=updatable_only, recurse=recurse):
                        yield f'{key}.{i}.{subkey}', subvalue
            elif recurse and info == 'dict':
                submodels: Dict[str, HealthVariable] = getattr(self, key)
                for subkey, submodel in submodels.items():
                    for subsubkey, subvalue in submodel.iter_parameters(updatable_only=updatable_only, recurse=recurse):
                        yield f'{key}.{subkey}.{subsubkey}', subvalue
            elif not recurse:
                pass  # All is good
            else:
                raise NotImplementedError(f'Unrecognized type for {key}: {info}')

    def get_parameters(self, names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Get updatable parameters as a numpy vector

        Args:
            names: Names of the parameters to gather. If ``None``, then will return all updatable parameters
        Returns:
            A numpy array of the values
        """

        # Get all variables if no specific list is specified
        if names is None:
            names = list(k for k, v in self.iter_parameters())

        # Special case, return an empty 2D array if no names were provided
        if len(names) == 0:
            return np.zeros((self.batch_size, 0))

        # Determine the batch dimension of the output
        batch_size = self.batch_size
        is_batched = batch_size > 1

        output = []
        for name in names:
            my_names, my_models = self._get_model_chain(name)
            value: np.ndarray = getattr(my_models[-1], my_names[-1])

            # Expand the array along batch and parameter dimension if needed
            if is_batched and value.shape[0] == 1:
                value = np.repeat(value, batch_size, axis=0)
            output.append(value)
        return np.concatenate(output, axis=1)  # Combine along the non-batched dimension

    def update_parameters(self, values: Union[np.ndarray, list[float]], names: Optional[Sequence[str]] = None):
        """
        Set the value for updatable parameters given their names

        Args:
            values: Values of the parameters to set
            names: Names of the parameters to set. If ``None``, then will set all updatable parameters
        """

        # Increase the shape to 2D
        values = np.array(values)
        if values.ndim == 1:
            values = values[None, :]

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
            num_params = cur_value.shape[1]  # Get the number of parameters

            # Get the parameters to use
            end = pos + num_params
            if pos + num_params > values.shape[-1]:
                raise ValueError(f'Required at least {end} values, but only provided {len(values)}')
            new_value = values[:, pos:end]
            setattr(model, attr, new_value)

            # Increment the starting point for the next parameter
            pos = end

        # Check to make sure all were used
        if end != values.shape[-1]:
            raise ValueError(f'Did not use all parameters. Provided {len(values)}, used {end}')

    def make_copy(self, values: np.ndarray, names: Optional[Sequence[str]] = None) -> Self:
        """
        Create a copy of the current object with values specified by numpy.ndarray

        Args:
            values: numpy array containing values to be used in copy
            names: sequence of the names of attributes to be returned with the values passed.
                If ``None``, changes all updatable parameters
        """
        copy = self.model_copy(deep=True)
        copy.update_parameters(values=values, names=names)
        return copy


class GeneralContainer(BaseModel,
                       arbitrary_types_allowed=True):
    """
    General container class to store numeric variables.

    Like the :class:`HealthVariable` all values are stored as 2d numpy arrays where the first dimension is a
    batch dimension. Accordingly, denote the types of attributes using the :class:`ScalarParameter` or
    :class:`ListParameter` for scalar and 1-dimensional data, respectively.
    """

    @cached_property
    def all_fields(self) -> tuple[str, ...]:
        """Names of all fields of the model in the order they appear in :meth:`to_numpy`

        Returns a single name per field, regardless of whether the field is a scalar or vector.
        See :meth:`all_names` to get a single name per value.
        """
        return tuple(self.__class__.model_fields.keys())

    @property
    def all_names(self) -> tuple[str, ...]:
        """Names of each value within the vector"""
        return tuple(self.expand_names(self.all_fields))

    def expand_names(self, names: Iterable[str]) -> tuple[str, ...]:
        """Expand a single name per field to a distinct name for each value within the field"""

        output = []
        for name in names:
            field: Optional[np.ndarray] = getattr(self, name)
            if field is None:
                continue
            length = field.shape[1]
            if length == 1:
                output.append(name)
            else:
                output.extend(f'{name}[{i}]' for i in range(length))
        return tuple(output)

    def __len__(self) -> int:
        """ Returns total length of all numerical values stored """
        return sum(self.length_field(field_name) for field_name in self.all_fields)

    @property
    def batch_size(self) -> int:
        """Batch size determined from the batch dimension of all attributes"""
        batch_size = 1
        batch_param_name = None  # Name of the parameter which is setting the batch size
        for name in self.all_fields:
            param = getattr(self, name)
            if param is None:
                continue

            my_batch = param.shape[0]
            if batch_size > 1 and my_batch != 1 and my_batch != batch_size:
                raise ValueError(f'Inconsistent batch sizes. {name} has batch dim of {my_batch},'
                                 f' whereas {batch_param_name} has a size of {batch_size}')
            batch_size = max(batch_size, my_batch)
            if my_batch == batch_size:
                batch_param_name = name
        return batch_size

    def length_field(self, field_name: str) -> int:
        """
        Returns length of provided field name. If the field is a float, returns 1, otherwise, returns length of array.
        If field is None, returns 0.
        """
        field_val: np.ndarray = getattr(self, field_name, None)
        if field_val is None:
            return 0
        return field_val.shape[-1]

    def to_numpy(self) -> np.ndarray:
        """
        Outputs everything that is stored as a two-dimensional np.ndarray
        """
        relevant_vals = []
        batch_size = self.batch_size
        for field_name in self.all_fields:
            field: Optional[np.ndarray] = getattr(self, field_name, None)
            if field is None:
                continue

            # Expand the batch dimension if needed
            if batch_size > 1 and field.shape[0] == 1:
                field = np.repeat(field, batch_size, 0)
            relevant_vals.append(field)
        return np.concatenate(relevant_vals, axis=1)

    def from_numpy(self, values: np.ndarray) -> None:
        """
        Updates field values from a numpy array
        """

        # Sure the values are a 2D array
        if values.ndim == 1:
            values = values[None, :]

        # We need to know where to start reading from in the array
        begin_index = 0
        for field_name in self.all_fields:
            current_field: Optional[np.ndarray] = getattr(self, field_name)
            if current_field is None:
                continue
            field_len = current_field.shape[-1]

            end_index = begin_index + field_len
            new_field_values = values[:, begin_index:end_index]
            setattr(self, field_name, new_field_values)
            begin_index = end_index

    def make_copy(self, values: np.ndarray) -> Self:
        """
        Helper method that returns a copy of the current object with values specified by numpy.ndarray

        Args:
            values: numpy array containing values to be used in copy
        """
        copy = self.model_copy(deep=True)
        copy.from_numpy(values)
        return copy


class InputQuantities(GeneralContainer):
    """
    The control of a battery system, such as the terminal current
    """
    time: ScalarParameter = Field(default=0., description='Timestamp(s) of inputs. Units: s')
    current: ScalarParameter = Field(default=0., description='Current applied to the storage system. Units: A')
    temperature: Optional[ScalarParameter] = Field(None, description='Temperature reading(s). Units: °C')


class OutputQuantities(GeneralContainer):
    """
    Output for observables from a battery system
    """

    terminal_voltage: ScalarParameter = \
        Field(description='Voltage output of a battery cell/model. Units: V')


class CellModel:
    """Base model for an energy storage system.

    Cell models describe how to update the transient state of a system and compute expected outputs
    given the inputs and current A-SOH.
    """

    @abstractmethod
    def update_transient_state(
            self,
            previous_inputs: InputQuantities,
            new_inputs: InputQuantities,
            transient_state: GeneralContainer,
            asoh: HealthVariable
    ) -> GeneralContainer:
        """
        Update the transient state of a chemical cell

        Args:
            previous_inputs: Inputs at the last time step
            new_inputs: Inputs at the current time step
            transient_state: Current transient state
            asoh: Health parameters of the cell

        Returns:
            A new transient state
        """
        pass

    @abstractmethod
    def calculate_terminal_voltage(
            self,
            new_inputs: InputQuantities,
            transient_state: GeneralContainer,
            asoh: HealthVariable) -> OutputQuantities:
        """
        Compute expected output (terminal voltage, etc.) of the cell.

        Args:
            new_inputs: Inputs at the current time step
            transient_state: Current transient state
            asoh: Health parameters of the cell
        Returns:
            Estimates for all measurable outputs of a cell
        """
        pass


class DegradationModel:
    """
    Base class for A-SOH aging models.

    Degradation models update the A-SOH incrementally given the current transient state,
    similar to how the :class:`CellModel` updates the transient state given current A-SOH.
    """

    @abstractmethod
    def update_asoh(self,
                    previous_asoh: HealthVariable,
                    new_inputs: InputQuantities,
                    new_transients: Optional[GeneralContainer],
                    new_measurements: Optional[OutputQuantities]) -> HealthVariable:
        """
        Degrade previous A-SOH based on inputs.

        Args:
            previous_asoh: previous A-SOH to be updated
            new_inputs: new inputs since the previous A-SOH
            new_transients: new transient states since the previous A-SOH
            new_measurements: new outputs since the previous A-SOH

        Returns:
            A new A-SOH object representing the degraded state
        """
        raise NotImplementedError("Please implement in child class!")
