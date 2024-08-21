"""Classes defining different multivariate probability distributions"""
from abc import abstractmethod
from warnings import warn
from typing import Iterable
from typing_extensions import Self

import numpy as np
from scipy.linalg import block_diag
from pydantic import Field, field_validator, computed_field, model_validator, BaseModel


class MultivariateRandomDistribution(BaseModel, arbitrary_types_allowed=True):
    """
    Base class to help represent a multivariate random variable
    """

    @computed_field
    @property
    def num_dimensions(self) -> int:
        """ Number of dimensions of random variable """
        return len(self.get_mean())

    @abstractmethod
    def get_mean(self) -> np.ndarray:
        """
        Provides mean (first moment) of distribution
        """
        raise NotImplementedError('Please implement in child class!')

    @abstractmethod
    def get_covariance(self) -> np.ndarray:
        """
        Provides the covariance of the distribution
        """
        raise NotImplementedError('Please implement in child class!')

    @abstractmethod
    def combine_with(self, random_dists: Iterable[Self]) -> Self:
        """
        Provides an easy way to combine several multivariate independent random variables of the same distribution type
        (delta, gaussian, etc.), but not necessarily of the same dimensions or from the same PDFs. It must not change
        self!
        """
        raise NotImplementedError('Please implement in child class!')


class DeltaDistribution(MultivariateRandomDistribution):
    """A distribution with only one allowed value"""

    mean: np.ndarray = Field(default=None, description='Mean of the distribution.')

    def get_mean(self) -> np.ndarray:
        return self.mean.copy()

    def get_covariance(self) -> np.ndarray:
        size = self.get_mean().shape[0]
        return np.zeros((size, size))

    def combine_with(self, random_dists: Iterable[Self]) -> Self:
        combined_mean = [self.get_mean(),]
        combined_mean += [delta.get_mean() for delta in random_dists]
        combined_mean = np.concatenate(combined_mean, axis=None)
        return DeltaDistribution(mean=combined_mean)


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

    def get_mean(self) -> np.ndarray:
        return self.mean.copy()

    def get_covariance(self) -> np.ndarray:
        return self.covariance.copy()

    def combine_with(self, random_dists: Iterable[Self]) -> Self:
        combined_mean = [self.get_mean(),]
        combined_mean += [gaussian.get_mean() for gaussian in random_dists]
        combined_cov = [self.get_covariance(),]
        combined_cov += [gaussian.get_covariance() for gaussian in random_dists]
        combined_mean = np.concatenate(combined_mean, axis=None)
        combined_cov = block_diag(*combined_cov)
        return MultivariateGaussian(mean=combined_mean, covariance=combined_cov)
