import numpy as np
from scipy.linalg import block_diag

from moirae.estimators.online.filters.distributions import MultivariateGaussian, DeltaDistribution


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
