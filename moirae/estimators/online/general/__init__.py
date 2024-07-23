""" Base online estimators, which are all model-agnostic """
from warnings import warn
from typing_extensions import Self

import numpy as np
from pydantic import Field, field_validator, computed_field, model_validator

from moirae.estimators.online import MultivariateRandomDistribution


class MultivariateGaussian(MultivariateRandomDistribution, validate_assignment=True):
    """
    Class to describe a multivariate Gaussian distribution
    """
    mean: np.ndarray = Field(default=np.array([0]),
                             description='Mean of the multivariate Gaussian distribution',
                             min_length=1)
    covariance: np.ndarray = Field(default=np.array([[1]]),
                                   description='Covariance of the multivariate Gaussian distribution',
                                   min_length=1)

    @field_validator('mean', mode='after')
    @classmethod
    def mean_1d(cls, mu: np.ndarray) -> np.ndarray:
        """ Making sure the mean is a vector """
        mean_shape = mu.shape
        if not mean_shape:
            raise ValueError('Mean must be Sized and have a non-empty shape!')
        if len(mean_shape) > 2 or (len(mean_shape) == 2 and 1 not in mean_shape):
            raise ValueError('Mean must be a 1D vector, but array provided has shape ' + str(mean_shape) + '!')
        elif len(mean_shape) == 2:
            msg = 'Provided mean has shape (%d, %d), but it will be flattened to (%d,)' % \
                (mean_shape + (max(mean_shape),))
            warn(msg)
        return mu.flatten()

    @field_validator('covariance', mode='after')
    @classmethod
    def cov_2d(cls, sigma: np.ndarray) -> np.ndarray:
        """ Making sure the covariance is a 2D matrix """
        cov_shape = sigma.shape
        if len(cov_shape) != 2:
            raise ValueError('Covariance must be a 2D matrix, but shape provided was ' + str(cov_shape) + '!')
        return sigma

    @model_validator(mode='after')
    def fields_dim(self) -> Self:
        """ Making sure dimensions match between mean and covariance """
        dim = self.num_dimensions
        if self.covariance.shape != (dim, dim):
            msg = 'Wrong dimensions! Mean has shape ' + str(self.mean.shape)
            msg += ', but covariance has shape ' + str(self.covariance.shape)
            raise ValueError(msg)
        return self

    @computed_field
    @property
    def num_dimensions(self) -> int:
        """ Number of dimensions of random variable """
        return len(self.mean)

    def get_mean(self) -> np.ndarray:
        return self.mean.copy()

    def get_covariance(self) -> np.ndarray:
        return self.covariance.copy()
