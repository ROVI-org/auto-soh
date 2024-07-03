import numpy as np
from pydantic import Field
from pytest import fixture

from asoh.models.base import HealthVariable


class SubHeathVariable(HealthVariable):
    """A sub-variable example used for testing"""

    x: float = -1.


class TestHealthVariable(HealthVariable):
    """A HealthVariable class which uses all types of allowed variables"""

    a: float = 1.
    b: np.ndarray = Field(default_factory=lambda: np.array([2.]))
    c: SubHeathVariable = Field(default_factory=SubHeathVariable)
    d: list[SubHeathVariable] = Field(default_factory=list)
    e: dict[str, SubHeathVariable] = Field(default_factory=dict)


@fixture
def example_hv() -> TestHealthVariable:
    return TestHealthVariable(
        d=[SubHeathVariable(x=0), SubHeathVariable(x=-2)],
        e={'first': SubHeathVariable(x=1)}
    )


def test_parameter_iterator(example_hv):
    """Test the iterator over all parameters"""

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
    assert np.isclose(parameters[1][1], [2.]).all()

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
