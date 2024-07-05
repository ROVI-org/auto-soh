""" Testing ECM utils """
import numpy as np
from pytest import fixture

from asoh.models.ecm.utils import SOCInterpolatedHealth


@fixture
def infer_soc() -> SOCInterpolatedHealth:
    return SOCInterpolatedHealth(base_values=np.linspace(0, 1, 10))


@fixture
def provide_soc() -> SOCInterpolatedHealth:
    return SOCInterpolatedHealth(base_values=np.linspace(0, 1, 10),
                                 soc_pinpoints=np.linspace(0, 1, 10))


@fixture
def constant() -> SOCInterpolatedHealth:
    return SOCInterpolatedHealth(base_values=np.pi)


@fixture
def linear_scale() -> SOCInterpolatedHealth:
    values = np.pi * np.linspace(0, 1, 10)
    return SOCInterpolatedHealth(base_values=values)


@fixture
def quadratic() -> SOCInterpolatedHealth:
    values = np.linspace(0, 1, 100) ** 2
    return SOCInterpolatedHealth(base_values=values,
                                 interpolation_style='quadratic')


@fixture
def cubic() -> SOCInterpolatedHealth:
    values = np.linspace(0, 1, 100) ** 3
    return SOCInterpolatedHealth(base_values=values,
                                 interpolation_style='cubic')


def test_inference(infer_soc):
    assert infer_soc.soc_pinpoints is None
    values = np.random.rand(100)
    assert np.isclose(values, infer_soc.get_value(values), atol=1e-12).all(), \
        'Wrong interpolation!'
    assert np.isclose(np.linspace(0, 1, 10), infer_soc.soc_pinpoints).all(), \
        'Wrong SOC inference!'


def test_provided(provide_soc):
    assert np.isclose(np.linspace(0, 1, 10),
                      provide_soc.soc_pinpoints,
                      atol=1e-12).all(), 'Wrong SOC setting!'
    values = np.random.rand(100)
    assert np.isclose(values, provide_soc.get_value(values), atol=1e-12).all(), \
        'Wrong interpolation!'


def test_constant(constant):
    values = np.random.rand(100)
    assert np.isclose([3.14159265358979323846264] * len(values),
                      constant.get_value(values),
                      atol=1e-12).all(), 'Wrong constant interpolation!'


def test_scale(linear_scale):
    values = np.random.rand(100)
    assert np.isclose(np.pi * values,
                      linear_scale.get_value(values),
                      atol=1e-12).all(), 'Wrong scaling interpolation!'


def test_quad(quadratic):
    values = np.random.rand(100)
    assert np.isclose(values * values,
                      quadratic.get_value(values),
                      atol=1e-12).all(), 'Wrong quadratic interpolation!'


def test_cube(cubic):
    values = np.random.rand(100)
    assert np.isclose(values * values * values,
                      cubic.get_value(values),
                      atol=1e-12).all(), 'Wrong cubic interpolation!'
