import numpy as np
from scipy.linalg import block_diag

from moirae.estimators.online.filters.distributions import MultivariateGaussian, DeltaDistribution
from moirae.estimators.online.filters.conversions import LinearConversionOperator


def test_combine():

    gaussian0 = MultivariateGaussian(mean=np.zeros(2), covariance=np.eye(2))
    gaussian1 = MultivariateGaussian(mean=np.ones(3), covariance=2 * np.eye(3))
    comb_gauss = gaussian0.combine_with(random_dists=[gaussian1])
    assert np.allclose(comb_gauss.get_mean(), [0., 0., 1., 1., 1.])
    assert np.allclose(comb_gauss.get_covariance(), block_diag(np.eye(2), 2 * np.eye(3)))

    delta0 = DeltaDistribution(mean=np.zeros(2))
    delta1 = DeltaDistribution(mean=np.ones(3))
    comb_delta = delta0.combine_with(random_dists=(delta1,))
    assert np.allclose(comb_delta.get_mean(), [0., 0., 1., 1., 1.])


def test_conversion():

    # Establish linear conversion
    multi = np.array([1, 2])
    bias = np.array(10)
    convertor = LinearConversionOperator(multiplicative_array=multi, additive_array=bias)

    # Delta
    delta = DeltaDistribution(mean=np.ones(2))
    transform_delta = delta.convert(conversion_operator=convertor)
    assert np.allclose(delta.get_mean(), np.ones(2)), 'Conversion changed base DeltaDist!!'
    assert np.allclose(transform_delta.get_mean(), [11, 12])

    # Gaussian
    gauss = MultivariateGaussian(mean=np.ones(2), covariance=np.diag([1, 2]))
    transform_gauss = gauss.convert(conversion_operator=convertor)
    assert np.allclose(gauss.get_mean(), np.ones(2)), 'Conversion changed base Gaussian mean!!'
    assert np.allclose(gauss.get_covariance(), np.diag([1, 2])), 'Conversion changed base Gaussian covariance!!'
    assert np.allclose(transform_gauss.get_mean(), np.array([11, 12]))
    assert np.allclose(transform_gauss.get_covariance(), np.diag([1, 8]))
