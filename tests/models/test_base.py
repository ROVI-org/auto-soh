import numpy as np
from pydantic import Field
from pytest import fixture, raises

from asoh.models.base import HealthVariable


class SubHeathVariable(HealthVariable):
    """A sub-variable example used for testing"""

    x: float = -1.


class ExampleHealthVariable(HealthVariable):
    """A HealthVariable class which uses all types of allowed variables"""

    a: float = 1.
    """A parameters"""
    b: np.ndarray = Field(default_factory=lambda: np.array([2., 3.]))
    c: SubHeathVariable = Field(default_factory=SubHeathVariable)
    d: tuple[SubHeathVariable, ...] = Field(default_factory=tuple)
    e: dict[str, SubHeathVariable] = Field(default_factory=dict)


@fixture
def example_hv() -> ExampleHealthVariable:
    return ExampleHealthVariable(
        d=[SubHeathVariable(x=0), SubHeathVariable(x=-2)],
        e={'first': SubHeathVariable(x=1)}
    )


def test_parameter_iterator(example_hv):
    """Test the iterator over parameters"""

    # Test iterating over all variables
    all_vars = list(example_hv.iter_parameters(updatable_only=False))
    assert len(all_vars) == 2 + 1 + 2 + 1  # 2 attributes for this field, 1 from c, 1 each from 2 in d, 1 for from in e

    top_vars = list(example_hv.iter_parameters(updatable_only=False, recurse=False))
    assert len(top_vars) == 2

    # Nothing should be iterable at the beginning
    assert example_hv.updatable == set()
    assert list(example_hv.iter_parameters()) == []

    # Test the non-recursive fields
    example_hv.updatable.update(['a', 'b'])
    assert example_hv.updatable == {'a', 'b'}

    parameters = list(example_hv.iter_parameters())
    assert len(parameters) == 2
    assert [k for k, _ in parameters] == ['a', 'b']
    assert np.isclose(parameters[0][1], [1.]).all()
    assert np.isclose(parameters[1][1], [2., 3.]).all()

    # Test the submodel when no fields of the submodel are updatable
    example_hv.updatable = {'c'}

    parameters = list(example_hv.iter_parameters())
    assert len(parameters) == 0

    # Make the fields updatable
    example_hv.c.make_all_updatable()
    assert example_hv.c.updatable == {'x'}

    parameters = list(example_hv.iter_parameters())
    assert len(parameters) == 1
    assert parameters[0][0] == 'c.x'

    # Test marking all fields as updatable non-recursively
    example_hv.make_all_updatable(recurse=False)
    assert example_hv.updatable == {'a', 'b', 'c', 'd', 'e'}

    parameters = dict(example_hv.iter_parameters())
    assert len(parameters) == 3  # d and e don't yet have any updatable parameters
    assert list(parameters.keys()) == ['a', 'b', 'c.x']

    # Now make _everything_ updatable
    example_hv.make_all_updatable()

    parameters = dict(example_hv.iter_parameters())
    assert len(parameters) == 6  # a, b, 1 from c, 2 from d, 1 from e
    assert np.isclose(parameters['c.x'], [-1.]).all()
    assert np.isclose(parameters['d.0.x'], [0.]).all()
    assert np.isclose(parameters['d.1.x'], [-2.]).all()
    assert np.isclose(parameters['e.first.x'], [1.]).all()


def test_get_model(example_hv):
    """Test getting the model which holds certain parameters.

    This will be used in the `get` and `update` methods"""

    # "is" tests whether it is not just equal, but _the same_ object
    assert example_hv._get_controling_model('a') is example_hv
    assert example_hv._get_controling_model('c.x') is example_hv.c
    assert example_hv._get_controling_model('d.0.x') is example_hv.d[0]
    assert example_hv._get_controling_model('e.first.x') is example_hv.e['first']


def test_set_value(example_hv):
    example_hv.set_value('a', 2.5)
    assert example_hv.a == 2.5

    example_hv.set_value('b', np.array([1., 2.]))
    assert np.isclose(example_hv.b, [1., 2.]).all()

    example_hv.set_value('e.first.x', -2.5)
    assert example_hv.e['first'].x == -2.5


def test_update_multiple_values(example_hv):
    example_hv.update_parameters(np.array([2.5, 1., 2., -2.5]), ['a', 'b', 'e.first.x'])
    assert example_hv.a == 2.5
    assert np.isclose(example_hv.b, [1., 2.]).all()
    assert example_hv.e['first'].x == -2.5

    # Make sure the error conditions work
    with raises(ValueError, match='^Did not use all'):
        example_hv.update_parameters(np.array([2.5, 1., 2., -2.5, np.nan]), ['a', 'b', 'e.first.x'])

    with raises(ValueError, match='^Required at least 4 values'):
        example_hv.update_parameters(np.array([2.5, 1., 2.]), ['a', 'b', 'e.first.x'])

    # Test setting only learnable parameters
    example_hv.updatable.add('c')
    example_hv.c.updatable.add('x')

    example_hv.update_parameters(np.array([-10.]))
    assert example_hv.c.x == -10.
