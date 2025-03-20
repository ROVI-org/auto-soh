import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

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


def test_inverse_conversion():

    # Establish linear conversion
    multi = np.array([1, 2])
    bias = np.array(10)
    convertor = LinearConversionOperator(multiplicative_array=multi, additive_array=bias)

    # Delta
    delta = DeltaDistribution(mean=np.ones(2))
    transform_delta = delta.convert(conversion_operator=convertor, inverse=True)
    assert np.allclose(delta.get_mean(), np.ones(2)), 'Conversion changed base DeltaDist!!'
    assert np.allclose(transform_delta.get_mean(), [-9, -4.5])

    # Gaussian
    gauss = MultivariateGaussian(mean=np.ones(2), covariance=np.diag([1, 2]))
    transform_gauss = gauss.convert(conversion_operator=convertor, inverse=True)
    assert np.allclose(gauss.get_mean(), np.ones(2)), 'Conversion changed base Gaussian mean!!'
    assert np.allclose(gauss.get_covariance(), np.diag([1, 2])), 'Conversion changed base Gaussian covariance!!'
    assert np.allclose(transform_gauss.get_mean(), np.array([-9, -4.5]))
    assert np.allclose(transform_gauss.get_covariance(), np.diag([1, 0.5]))


def test_log_likelihood():

    # Data to compute
    data = np.array([[0., 0.],
                     [5., 6.],
                     [6., 5.],
                     [10., 12.]])

    # Delta
    delta = DeltaDistribution(mean=np.array([5., 6.]))
    delta_log_like = delta.compute_log_likelihook(data=data)
    expected = -np.inf * np.ones(len(data))
    expected[1] = 0.
    assert np.allclose(delta_log_like, expected), \
        f'Wrong Delta log-likelihood! Expected {expected}, got {delta_log_like}'

    # Gaussian
    mean = data[1, :]
    cov = np.eye(2)  # independent for now
    ind_gauss = MultivariateGaussian(mean=mean, covariance=cov)
    # Create a similar one with scipy stats
    mv_ind_gauss = multivariate_normal(mean=mean, cov=cov)
    ind_log_like = ind_gauss.compute_log_likelihook(data=data)
    assert np.allclose(ind_log_like[1], -np.log(2 * np.pi)), f'Unexpected likelihood for the mean: {ind_log_like[1]}!'
    assert ind_log_like[0] == ind_log_like[-1], f'Independent Gaussian likelihood not symmetric: {ind_log_like}'
    assert np.allclose(ind_log_like, np.log(mv_ind_gauss.pdf(data))), f'{mv_ind_gauss.pdf(data)}'
    # Now, consider one in which the dimensions are not independent
    cov = np.array([[1, -0.5], [-0.5, 2]])
    ind_gauss = MultivariateGaussian(mean=mean, covariance=cov)
    # Create a similar one with scipy stats
    mv_cor_gauss = multivariate_normal(mean=mean, cov=cov)
    cor_log_like = ind_gauss.compute_log_likelihook(data=data)
    mean_likelihood = -np.log(2 * np.pi) - (0.5 * np.log(1.75))
    assert np.allclose(cor_log_like[1], mean_likelihood), f'Unexpected likelihood for the mean: {cor_log_like[1]}!'
    assert cor_log_like[0] == cor_log_like[-1], f'Correlated Gaussian likelihood not symmetric: {cor_log_like}'
    assert np.allclose(cor_log_like, np.log(mv_cor_gauss.pdf(data))), f'{mv_cor_gauss.pdf(data)}'
