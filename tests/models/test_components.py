""" Testing ECM utils """
import numpy as np
from pytest import fixture, mark

from moirae.models.components.soc import SOCInterpolatedHealth, ScaledSOCInterpolatedHealth, SOCPolynomialHealth
from moirae.models.components.soc_t import SOCTempPolynomialHealth


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
    result = constant.get_value(values)
    assert result.shape == (1, 100)
    assert np.allclose([3.14159265358979323846264], result,
                       atol=1e-12), 'Wrong constant interpolation!'

    # Test with batching the SOCs
    result = constant.get_value(values[:, None])
    assert result.shape == (100, 1)

    # Test with batching the base values
    constant.base_values = np.array([[1], [2]])
    result = constant.get_value(1.)
    assert result.shape == (2, 1)
    assert np.allclose(result, constant.base_values)

    # Test with getting only a certain batch member
    result = constant.get_value(1., batch_id=1)
    assert result.shape == (1, 1)
    assert np.allclose(2, result)


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


def test_batching_interp(cubic):
    """Ensure that we get the correct answers with different batch conditions"""
    # Batched SOC, singular params
    socs = np.array([[0.1], [0.4]])
    result = cubic.get_value(socs)
    assert result.shape == (2, 1)
    assert np.allclose(socs ** 3, result)

    result = cubic.get_value(socs, batch_id=1)
    assert result.shape == (1, 1)
    assert np.allclose(0.4 ** 3, result)

    # Singular SOC, batched params
    cubic.base_values = cubic.base_values * np.array([[2], [1]])
    assert cubic.batch_size == 2
    socs = np.array([[0.1]])

    result = cubic.get_value(socs)
    assert result.shape == (2, 1)
    assert np.allclose(2 * socs ** 3, result[0, 0])
    assert np.allclose(socs ** 3, result[1, 0])

    result = cubic.get_value(socs, batch_id=1)
    assert result.shape == (1, 1)
    assert np.allclose(0.1 ** 3, result)


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
    assert np.allclose(variable.soc_pinpoints, re_variable0.soc_pinpoints), 'Wrong recreation of SOC pinpoints!'


@mark.parametrize('inter_batch,scale_batch,additive',
                  [(1, 1, True), (1, 2, True), (2, 1, True), (2, 2, True),
                   (2, 2, False)])
def test_scaling(inter_batch, scale_batch, additive):
    """Test the SOC interpolation with a Legendre scaling factor"""

    # Make the scaling tool
    soc = np.linspace(0, 1, 9)
    inter_values = np.linspace(0, 1, 10)[None, :] * (np.arange(inter_batch) + 1)[:, None]
    scaling_values = np.array([[0.001]]) * (np.arange(scale_batch) + 1)[:, None]
    unscaled = SOCInterpolatedHealth(base_values=inter_values)
    scaled = ScaledSOCInterpolatedHealth(
        base_values=inter_values,
        scaling_coeffs=scaling_values,
        additive=additive
    )

    # Check the basics: shape and that _something_ changed
    scaled_val = scaled.get_value(soc)
    unscaled_val = unscaled.get_value(soc)
    assert not np.allclose(scaled_val, unscaled_val)
    assert scaled_val.shape == (max(inter_batch, scale_batch), 9)

    # Check the changed amount
    if additive:
        assert np.allclose(scaled_val[0, :] - unscaled_val[0, :], 0.001)
        if scale_batch == 2:
            assert np.allclose(scaled_val[1, :] - unscaled_val[1 % inter_batch, :], 0.002)
    else:
        assert np.allclose(scaled_val[0, 1:] / unscaled_val[0, 1:], 1.001)

    # Make sure it works with selecting a specific ID
    single_id = scaled.get_value(soc, batch_id=0)
    assert single_id.shape == (1, scaled_val.shape[1])
    assert np.allclose(single_id, scaled_val[0, :])


def test_soc_polynomial():
    comp = SOCPolynomialHealth(coeffs=[[1, 0.5], [0.5, 0.25]])

    # Make sure it works with scalars for a single batch, as needed by Thevenin to update
    y = comp.get_value(0.5, batch_id=0)
    assert y.shape == (1, 1)
    assert np.isclose(y, 1.25)

    y = comp.get_value(0.5, batch_id=1)
    assert y.shape == (1, 1)
    assert np.isclose(y, 0.625)

    # Make sure it works with 2D inputs and all batches, as when computing terminal voltage
    y = comp.get_value(np.array([0.5]))
    assert y.shape == (2, 1)
    assert np.allclose(y, [[1.25], [0.625]])

    y = comp.get_value(np.array([[0.5], [0.]]))  # Evaluate different SOC points for different batch members
    assert y.shape == (2, 1)
    assert np.allclose(y, [[1.25], [0.5]])


def test_soc_temp_dependence():
    comp = SOCTempPolynomialHealth(soc_coeffs=[[1, 0.5], [0.5, 0.25]], t_coeffs=[[0, -0.1]])

    # Make sure it works with scalars for a single batch, as needed by Thevenin to update
    y = comp.get_value(0.5, 26, batch_id=0)
    assert y.shape == (1, 1)
    assert np.isclose(y, 1.15)

    y = comp.get_value(0.5, 26, batch_id=1)
    assert y.shape == (1, 1)
    assert np.isclose(y, 0.525)

    # Make sure it works with 2D inputs and all batches, as when computing terminal voltage
    y = comp.get_value(np.array([0.5]), np.array([26]))
    assert y.shape == (2, 1)
    assert np.allclose(y, [[1.15], [0.525]])

    y = comp.get_value(np.array([[0.5], [0.]]), np.array([26]))
    assert y.shape == (2, 1)
    assert np.allclose(y, [[1.15], [0.4]])
