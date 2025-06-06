""" Tools to convert between coordinate systems """
from abc import abstractmethod
from functools import cached_property
from typing import Optional, Literal, List

import numpy as np
from pydantic import BaseModel, Field, field_validator, computed_field


class ConversionOperator(BaseModel, arbitrary_types_allowed=True):
    """
    Base class used to convert between :class:`~moirae.models.base.CellModel` and
    :class:`~moirae.estimators.online.filters.base.BaseFilter` coordinate systems.

    Each implementation provides the ability to convert both points and covariances,
    as required to map :meth:`~moirae.estimators.online.filters.distributions.MultivariateRandomDistribution`
    between the coordinate systems employed by :class:`~moirae.estimators.online.filters.base.BaseFilter`
    and :class:`~moirae.models.base.CellModel`.
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
    def transform_covariance(self, covariance: np.ndarray, pivot: np.ndarray) -> np.ndarray:
        """
        Transforms the covariance of a
        :class:`~moirae.estimators.online.filters.distributions.MultivariateRandomDistribution`.

        Args:
            covariance: 2D np.ndarray corresponding to the covariance to be transformed
            pivot: central value around which covariance must be propagated (akin to first moment of distribution)

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
    def inverse_transform_covariance(self,
                                     transformed_covariance: np.ndarray,
                                     transformed_pivot: np.ndarray) -> np.ndarray:
        """
        Performs the inverse transformation of that given by
        :meth:`~moirae.estimators.online.filters.transformations.BaseTransform.transform_covariance`.

        Args:
            transformed_covariance: np.ndarray of transformed covariance to be converted back
            transformed_pivot: in cases where Taylor-expansion is employed, it should be performed around this
                transformed pivot point

        Returns:
            covariance: np.ndarray corresponding to re-converted covariance
        """
        raise NotImplementedError('Implement in child class!')


class IdentityConversionOperator(ConversionOperator):
    """
    Class that implements simple identity operation, that is, it does not change the inputs.
    """

    # DEV NOTE: Returning copies so that we are assured the inputs are unmodified
    #  regardless of how we use the output
    def transform_samples(self, samples: np.ndarray) -> np.ndarray:
        return samples.copy()

    def transform_covariance(self, covariance: np.ndarray, pivot: Optional[np.ndarray] = None) -> np.ndarray:
        return covariance.copy()

    def inverse_transform_samples(self, transformed_samples: np.ndarray) -> np.ndarray:
        return transformed_samples.copy()

    def inverse_transform_covariance(self,
                                     transformed_covariance: np.ndarray,
                                     transformed_pivot: Optional[np.ndarray] = None) -> np.ndarray:
        return transformed_covariance.copy()


class LinearConversionOperator(ConversionOperator):
    """
    A linear transformation, :math:`y = (multi * x) + basis`

    Args:
        multiplicative_array: np.ndarray corresponding to multiplicative factors in the linear function
        additive_array: np.ndarray corresponding to additive factors in the linear function
    """
    multiplicative_array: np.ndarray = Field(description='Multiplicative factors of linear function',
                                             default=np.array(1.))
    additive_array: Optional[np.ndarray] = Field(description='Additive factors of linear function',
                                                 default=np.array([0.]))

    @field_validator('additive_array', mode='after')
    @classmethod
    def additive_1d(cls, bias: np.ndarray) -> np.ndarray:
        """
        Ensures the additive array is a 1D vector, and not a matrix
        """
        if bias.ndim > 1:
            raise ValueError(f'Additive factor must be a vector or scalar, but, instead, is has shape {bias.ndim}!')
        return bias.flatten()

    @field_validator('multiplicative_array', mode='after')
    @classmethod
    def multiplicative_2d(cls, multi: np.ndarray) -> np.ndarray:
        """
        Ensures the multiplicative array is stored as a 2D array (or as a shapeless object)
        """
        if multi.ndim > 2:
            raise ValueError(f'Additive factor must be at most 2D, but, instead, is has shape {multi.ndim}!')
        if multi.ndim == 1:
            return np.diag(multi)
        return multi

    @computed_field
    @property
    def _len_multi_shape(self) -> Literal[0, 2]:
        return self.multiplicative_array.ndim

    @cached_property
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

    def transform_covariance(self, covariance: np.ndarray, pivot: Optional[np.ndarray] = None) -> np.ndarray:
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

    def inverse_transform_covariance(self,
                                     transformed_covariance: np.ndarray,
                                     transformed_pivot: Optional[np.ndarray] = None) -> np.ndarray:
        if self._len_multi_shape == 0:
            return self.inv_multi * self.inv_multi * transformed_covariance
        covariance = np.matmul(np.matmul(self.inv_multi.T, transformed_covariance), self.inv_multi)
        return covariance


class FirstOrderTaylorConversionOperator(ConversionOperator):
    """
    Base class that specifies the necessary machinery to perform non-linear conversions assuming a first order Taylor
    expansion around a pivot point is sufficient to propagate uncertainties (through covariances).

    Assumes the transformation can be expressed as :math:`f = f_0 + J (x-p)`, where :math:`f_0` represents the value of
    the transformation at the pivot point :math:`p`, and :math:`J` is the Jacobian matrix at the pivot. Based on this,
    the covariance of the transformed vector :math:`f` can be simply expressed as
    :math:`{\\Sigma}_f = J {\\Sigma}_X J^T`, exactly like that in the
    :class:`~moirae.estimators.online.filters.conversions.LinearConversionOperator`
    Full explanation on `Wikipedia <https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Non-linear_combinations>`_.
    """

    @abstractmethod
    def get_jacobian(self, pivot: np.ndarray) -> np.ndarray:
        """
        Method to calculate the Jacobian matrix of the transformation at specified pivot point

        Args:
            pivot: point to be used as the pivot of the Taylor expansion

        Returns:
            jacobian: Jacobian matrix
        """
        raise NotImplementedError('Implement in child class!')

    @abstractmethod
    def get_inverse_jacobian(self, transformed_pivot: np.ndarray) -> np.ndarray:
        """
        Method to compute the Jacobian matrix of the inverse transformation.

        Args:
            transformed_pivot: point to be used as the transformed pivot

        Returns:
            inverse_jacobian: Jacobian matrix of the reverse transformation
        """
        raise NotImplementedError('Implement in child function!')

    def transform_covariance(self, covariance: np.ndarray, pivot: np.ndarray) -> np.ndarray:
        jacobian = self.get_jacobian(pivot=pivot)
        return np.matmul(np.matmul(jacobian.T, covariance), jacobian)

    def inverse_transform_covariance(self,
                                     transformed_covariance: np.ndarray,
                                     transformed_pivot: np.ndarray) -> np.ndarray:
        inv_jacobian = self.get_inverse_jacobian(transformed_pivot=transformed_pivot)
        return np.matmul(np.matmul(inv_jacobian.T, transformed_covariance), inv_jacobian)


class AbsoluteValueConversionOperator(FirstOrderTaylorConversionOperator):
    """
    Class that performs an absolute value transformation to the samples.

    This is particularly relevant in cases where the estimators used treat values as unbounded, but the underlying
    models (such as an equivalent circuit) only accept positive parameters. Since the absolute value function is not
    invertible, the following choice was made for its inverse operations: the inverted values always belong to the
    positive quadrant of space, that is, they are always positive.

    Args:
        indices: indices to which the absolute value transformation should be applied; if not provided, applies absolute
            value to all values
    """
    indices: Optional[List[int]] = Field(default=None, description='indices to apply transformation')

    def get_jacobian(self, pivot: np.ndarray) -> np.ndarray:
        # The derivatives are equal to 1 where the pivot value is postive, and -1 otherwise
        diagonal = np.ones(pivot.size)
        if self.indices is None:
            diagonal = np.where(pivot >= 0, 1., -1.)
        else:
            diagonal[self.indices] = np.where(pivot[self.indices] >= 0, 1., -1.)
        return np.diag(diagonal)

    def get_inverse_jacobian(self, transformed_pivot: np.ndarray) -> np.ndarray:
        # This transformation is not invertible, so, to simplify, we will assume the conversion came from the positive
        # quadrant (that is, where all values, of the original and transformed pivot, are positive)
        return np.eye(transformed_pivot.size)

    def transform_samples(self, samples: np.ndarray) -> np.ndarray:
        if self.indices is None:
            transformed_samples = np.abs(samples)
        else:
            transformed_samples = samples.copy()
            if samples.ndim == 1:
                transformed_samples[self.indices] = np.abs(samples[self.indices])
            else:
                transformed_samples[:, self.indices] = np.abs(samples[:, self.indices])
        return transformed_samples

    def inverse_transform_samples(self, transformed_samples: np.ndarray) -> np.ndarray:
        # Once again, assume transformation goes positive quadrant <=> positive quadrant
        return transformed_samples.copy()
