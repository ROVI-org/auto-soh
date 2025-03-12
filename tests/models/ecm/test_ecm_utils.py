""" Testing ECM utils """
import numpy as np
from pytest import fixture

from moirae.models.ecm.utils import SOCInterpolatedHealth


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
    result = cubic.get_value(values)
    assert np.isclose(values * values * values, result,
                      atol=1e-12).all(), 'Wrong cubic interpolation!'
    cached_result = cubic.get_value(values)
    assert np.allclose(result, cached_result)


def test_serialization():
    # Create a variable
    variable = SOCInterpolatedHealth(base_values=np.arange(10))
    assert variable.soc_pinpoints is None, 'SOC pinpoints created prematurely!'

    # Make sure the base_values serialize correctly
    varialb_str0 = variable.model_dump_json()
    re_variable0 = SOCInterpolatedHealth.model_validate_json(varialb_str0)
    assert np.allclose(variable.base_values, re_variable0.base_values), 'Wrong recreation of base values!'
    assert re_variable0.soc_pinpoints is None, 'SOC points where created on serialization'

    # Now let's say we try to get a value, in which case, the soc_pinpoints are created
    variable.get_value(soc=0.5)
    assert len(variable.soc_pinpoints) == variable.base_values.shape[1], 'Mismatch between pinpoints and base values!'
    assert np.allclose(re_variable0.get_value(0.5), variable.get_value(soc=0.5)), 'Wrong calculation of value!'

    # Ensure the base values get serialized correctly
    varialb_str1 = variable.model_dump_json()
    re_variable1 = SOCInterpolatedHealth.model_validate_json(varialb_str1)
    assert np.allclose(variable.base_values, re_variable1.base_values), 'Wrong recreation of base values!'
    assert np.allclose(variable.soc_pinpoints, re_variable1.soc_pinpoints), 'Wrong recreation of SOC pinpoints!'

    # Now, try pre-defining soc pinpoints
    variable = SOCInterpolatedHealth(base_values=np.arange(10), soc_pinpoints=np.linspace(0, 1, 10))
    assert len(variable.soc_pinpoints) == 10, 'SOC pinpoints defined incorrectly!'

    # The usual stuff
    varialb_str0 = variable.model_dump_json()
    re_variable0 = SOCInterpolatedHealth.model_validate_json(varialb_str0)
    assert np.allclose(variable.base_values, re_variable0.base_values), 'Wrong recreation of base values!'
    assert np.allclose(variable.soc_pinpoints, re_variable0.soc_pinpoints), 'Wrong recreationg of SOC pinpoints!'
