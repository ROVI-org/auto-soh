import numpy as np
from pytest import raises

from moirae.estimators.offline.assemblers.utils import SOCRegressor
from moirae.estimators.offline.assemblers.ecm.utils import SOCDependentAssembler


def test_incomplete_soc_range():
    with raises(ValueError, match="SOC domain not fully covered, only 10.0 -- 100.0%!"):
        _ = SOCDependentAssembler(regressor=SOCRegressor(),
                                  soc_points=[0.1, 0.5, 1.])
    with raises(ValueError, match="SOC domain not fully covered, only 0.0 -- 90.0%!"):
        _ = SOCDependentAssembler(regressor=SOCRegressor(),
                                  soc_points=[0., 0.5, 0.9])
    with raises(ValueError, match="SOC domain not fully covered, only 10.0 -- 90.0%!"):
        _ = SOCDependentAssembler(regressor=SOCRegressor(),
                                  soc_points=[0.1, 0.5, 0.9])


def test_correct_soc_creation():
    assembler = SOCDependentAssembler(regressor=SOCRegressor(),
                                      soc_points=11)
    assert np.allclose(np.linspace(0, 1, 11), assembler.soc_points), f'Wrong SOC points created: {assembler.soc_points}'
