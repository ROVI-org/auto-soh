import numpy as np
from pytest import fixture
from pydantic import Field

from moirae.models.base import HealthVariable, ListParameter, ScalarParameter
from moirae.models.utils import DummyDegradation


class SubHeathVariable(HealthVariable):
    """A sub-variable example used for testing"""

    x: ScalarParameter = -1.


class ExampleHealthVariable(HealthVariable):
    """A HealthVariable class which uses all types of allowed variables"""

    a: ScalarParameter = 1.
    """A parameter"""
    b: ListParameter = Field(default_factory=lambda: [2., 3.])
    c: SubHeathVariable = Field(default_factory=SubHeathVariable)
    d: tuple[SubHeathVariable, ...] = Field(default_factory=tuple)


@fixture
def example_hv() -> HealthVariable:
    """
    Creates a very simple health variable
    """
    d0 = SubHeathVariable(x=5)
    d1 = SubHeathVariable(x=-5)
    ex = ExampleHealthVariable(d=(d0, d1))
    ex.mark_all_updatable()
    return ex


def test_dummy_deg(example_hv) -> None:
    deg_model = DummyDegradation()
    deg_ex = deg_model.update_asoh(previous_asoh=example_hv)
    assert np.allclose(example_hv.get_parameters(), deg_ex.get_parameters())
