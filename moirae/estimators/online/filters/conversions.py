""" Collection of base coordinate transformations"""
from abc import abstractmethod
from typing import Optional, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, computed_field


class ConversionOperator(BaseModel, arbitrary_types_allowed=True):
    """
    Base class used to convert between :class:`~moirae.models.base.CellModel` and
    :class:`~moirae.estimators.online.filters.base.BaseFilter` coordinate systems.

    Specifically, it is given to the
    :meth:`~moirae.estimators.online.filters.distributions.MultivariateRandomDistribution.transform` method of
    :class:`~moirae.estimators.online.filters.distributions.MultivariateRandomDistribution` to tell it how it should be
    transformed. It is used at the :class:`~moirae.estimators.online.filters.base.ModelWrapper` to help translate the
    output from :class:`~moirae.estimators.online.filters.base.BaseFilter` to numerical values that are pertinent to
    :class:`~moirae.models.base.CellModel`.
    """

    @abstractmethod
    def transform_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Transforms a set of individual points, considered to be independent samples of the same
        :class:`~moirae.estimators.online.filters.distributions.MultivariateRandomDistribution`.

        Args:
            samples: np.ndarray of samples to be transformed. Its shape must be (num_samples, num_coordinates) or
                (num_coordinates,)

        Returns:
            transformed_samples: np.ndarray of the transformed points
        """
        raise NotImplementedError('Implement in child class!')

    @abstractmethod
    def transform_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """
        Transforms the covariance of a
        :class:`~moirae.estimators.online.filters.distributions.MultivariateRandomDistribution`.

        Args:
            covariance: 2D np.ndarray corresponding to the covariance to be transformed

        Returns:
            transformed_covariance: transformed covariance matrix
        """
        raise NotImplementedError('Implement in child class!')

    @abstractmethod
    def inverse_transform_samples(self, transformed_samples: np.ndarray) -> np.ndarray:
        """
        Performs the inverse tranformation of that given by
        :meth:`~moirae.estimators.online.filters.transformations.BaseTransform.transform_points`.

        Args:
            transformed_samples: np.ndarray of points to be inverse transformed

        Returns:
            samples: np.ndarray corresponding to re-converted points
        """
        raise NotImplementedError('Implement in child class!')

    @abstractmethod
    def inverse_transform_covariance(self, transformed_covariance: np.ndarray) -> np.ndarray:
        """
        Performs the inverse tranformation of that given by
        :meth:`~moirae.estimators.online.filters.transformations.BaseTransform.transform_covariance`.

        Args:
            transformed_covariance: np.ndarray of transformed covariance to be converted back

        Returns:
            covariance: np.ndarray corresponding to re-converted covariance
        """
        raise NotImplementedError('Implement in child class!')


class LinearConversionOperator(ConversionOperator, validate_assignment=True):
    """
    Class that implements a linear function as a transformation (strictly speaking, this is not a linear transformation,
    but just a linear function).

    Given an array of multiplicative factors ``multi`` and additive factors ``bias``, this function transforms points
    ``x`` to ``y`` following ``y = (multi * x) + bias``

    Args:
        multiplicative_array: np.ndarray corresponding to multiplicative factors in the linear function
        additive_array: np.ndarray corresponding to additive (bias) factors in the linear function
    """
    multiplicative_array: np.ndarray = Field(description='Multiplicative factors of linear function',
                                             default=np.array(1.))
    additive_array: Optional[np.ndarray] = Field(description='Additive (bias) factors of linear function',
                                                 default=np.array([0.]))

    @field_validator('additive_array', mode='after')
    @classmethod
    def additive_1d(cls, bias: np.ndarray) -> np.ndarray:
        """
        Ensures the additive array is a 1D vector, and not a matrix
        """
        shape = bias.shape
        if len(shape) > 1:
            raise ValueError(f'Additive factor must be a vector or scalar, but, instead, is has shape {shape}!')
        return bias.flatten()

    @field_validator('multiplicative_array', mode='after')
    @classmethod
    def multiplicative_2d(cls, multi: np.ndarray) -> np.ndarray:
        """
        Ensures the multiplicative array is stored as a 2D array (or as a shapeless object)
        """
        shape = multi.shape
        if len(shape) > 2:
            raise ValueError(f'Additive factor must be at most 2D, but, instead, is has shape {shape}!')
        if len(shape) == 1:
            return np.diag(multi)

    @computed_field
    @property
    def _len_multi_shape(self) -> Literal[0, 2]:
        return len(self.multiplicative_array.shape)

    @computed_field
    @property
    def inv_multi(self) -> np.ndarray:
        """
        Stores the inverse of the multiplicative array, which is needed for the inverse operations
        """
        if self._len_multi_shape == 0:
            return 1 / self.multiplicative_array
        # There is a possibility the transformation is a dimensionality reduction, and the multiplicative array is not
        # a square matrix. Therefore, we will use the pseudo-inverse of the matrix (which is equal to the inverse in the
        # case of a square matrix): the pseudo-inverse is a right-inverse; for a matrix of shape ``(D,d)``, it outputs
        # an array of shape ``(d,D)``
        # NOTE: in case of dimensionality reduction (``multi.shape = (D,d)`` with ``D>d``),
        #   ``np.matmul(multi, np.linalg.pinv(multi))`` can be very far from identity!!
        return np.linalg.pinv(self.multiplicative_array)

    def transform_samples(self, samples: np.ndarray) -> np.ndarray:
        if self._len_multi_shape == 0:
            return (self.multiplicative_array * samples) + self.additive_array
        transformed_samples = np.matmul(samples, self.multiplicative_array) + self.additive_array
        return transformed_samples

    def transform_covariance(self, covariance: np.ndarray) -> np.ndarray:
        if self._len_multi_shape == 0:
            return self.multiplicative_array * self.multiplicative_array * covariance
        transformed_covariance = np.matmul(np.matmul(self.multiplicative_array.T, covariance),
                                           self.multiplicative_array)
        return transformed_covariance

    def inverse_transform_samples(self, transformed_samples: np.ndarray) -> np.ndarray:
        samples = transformed_samples - self.additive_array
        if self._len_multi_shape == 0:
            return samples * self.inv_multi
        samples = np.matmul(samples, self.inv_multi)
        return samples

    def inverse_transform_covariance(self, transformed_covariance: np.ndarray) -> np.ndarray:
        if self._len_multi_shape:
            return self.inv_multi * self.inv_multi * transformed_covariance
        covariance = np.matmul(np.matmul(self.inv_multi.T, transformed_covariance), self.inv_multi)
        return covariance
