from typing import Union, Tuple
from numbers import Number

import numpy as np
from pydantic import BaseModel, computed_field


class BatchedVariable(BaseModel,
                      arbitrary_types_allowed=True,
                      validate_assignment=True):
    """
    Class to help storing variables that can be batched. It effectively serves as a wrapper for both floats and numpy
    arrays, and makes it easier to figure out the batch size and inner dimensionality of the variable (in case of a
    multi-valued variable)

    Args:
        batched_values: value(s) of the variable at the differente batch(es)
        batch_size: cardinality of the batch
        inner_dimensions: number of dimensions of the variable
    """
    batched_values: Union[Number, np.ndarray]
    batch_size: int = 1
    inner_dimensions: int = 1

    @computed_field
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.batch_size, self.inner_dimensions)


def convert_single_valued(value: Union[float, np.ndarray]) -> BatchedVariable:
    """
    Helper function to convert (possibly batched) single-valued variable into appropriate BatchedVariable object
    """
    if isinstance(value, Number):
        return BatchedVariable(batched_values=np.array(value))
    elif len(value.shape) <= 2:
        if len(value.shape) == 2 and (1 not in value.shape):
            raise ValueError(f'Single-valued variable cannot be passed as a {value.shape} matrix; ',
                             'one of the dimensions must be equal to 1!')
        value = value.reshape((-1, 1))
        return BatchedVariable(batched_values=value, batch_size=value.shape[0])
    else:
        raise ValueError(f'Single-valued variable cannot be passed as >2D array; shape provided was {value.shape}')


def convert_multi_valued(values: np.ndarray) -> BatchedVariable:
    """
    Helper function to convert (possibly batched) multi-valued variable into appropriate BatchedVariable object
    """
    if len(values.shape) == 1:  # assume these are the inner dimensions
        return BatchedVariable(batched_values=np.atleast_2d(values), inner_dimensions=values.shape[0])
    elif len(values.shape) == 2:  # assume first axis is batch, second is inner
        return BatchedVariable(batched_values=values, batch_size=values.shape[0], inner_dimensions=values.shape[1])
    else:
        raise ValueError(f'Multi-valued variable cannot be passed as >2D array; shape provided was {values.shape}')
