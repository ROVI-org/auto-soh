from pytest import raises

from moirae.estimators.offline.assemblers.utils import SOCRegressor


def test_wrong_init():
    # Testing wrong style
    with raises(ValueError, match="Accepted fitting styles are"):
        _ = SOCRegressor(style='splin')

    # Test wrong parameter type
    with raises(TypeError, match="Parameters must be provided as a dictionary!"):
        _ = SOCRegressor(parameters=[1, 2, 3])

    # Test wrong parameters
    # Simple Interpolation
    mess = 'Acceptable parameters for interpolate are k, t, bc_type, axis, check_finite, '
    mess += 'not abcd!'
    with raises(ValueError, match=mess):
        _ = SOCRegressor(parameters={'abcd': 1234})
    # Spline
    mess = 'Acceptable parameters for smooth are lam, axis, '
    mess += 'not abcd!'
    with raises(ValueError, match=mess):
        _ = SOCRegressor(style='smooth', parameters={'abcd': 1234})
    # LSQ
    mess = 'Acceptable parameters for lsq are k, w, axis, check_finite, method, '
    mess += 'not abcd!'
    with raises(ValueError, match=mess):
        _ = SOCRegressor(style='lsq', parameters={'abcd': 1234})
    # Isotonic
    mess = 'Acceptable parameters for isotonic are y_min, y_max, increasing, out_of_bounds, '
    mess += 'not abcd!'
    with raises(ValueError, match=mess):
        _ = SOCRegressor(style='isotonic', parameters={'abcd': 1234})
    # Polyfit
    mess = 'Acceptable parameters for polyfit are deg, rcond, full, w, cov, '
    mess += 'not abcd!'
    with raises(ValueError, match=mess):
        _ = SOCRegressor(style='polyfit', parameters={'abcd': 1234})
