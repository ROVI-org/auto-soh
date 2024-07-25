from typing import Union
from numbers import Number

import numpy as np

from .base import BatchedVariable


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
