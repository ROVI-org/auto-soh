"""Interface for :class:`~moirae.estimators.online.filters.BaseFilter` so that they
can use a model wrapped by the :class:`~moirae.estimators.online.model.transformed.TransformedModel`."""
from typing import Tuple

from moirae.estimators.online import MultivariateRandomDistribution
from moirae.estimators.online.filters.base import BaseFilter
from moirae.estimators.online.filters.distributions import MultivariateGaussian
from moirae.estimators.online.model.transformed import TransformedModel


class TransformedFilter(BaseFilter):
    """Utility to rectify outputs from filters which different coordinate system than the underlying model.

    A partner to the :class:`~moirae.estimators.online.model.transformed.TransformedModel`.
    Supply a filter configured to work in a coordinate system of choice and a wrapped
    version of the model which can translate between the filter's coordinate system and that
    of the original model.

    The ``TransformedFilter`` alters the probability distribution for the hidden states returned
    by the wrapped filter into the same coordinate system used by the cell model,
    allowing any users of the filter to n

    Args:
        model: Model to used to describe the underlying physics and convert from the coordinate system
            used by the wrapped filter
        wrapped_filter: Filter which operates using an altered coordinate system
    """

    model: TransformedModel
    """Model that includes the logic to go to and from an altered coordinate system"""

    def __init__(self, model: TransformedModel, wrapped_filter: BaseFilter):
        if not isinstance(model, TransformedModel):
            raise ValueError(f'A transformed filter requires a transformed model, not a {model.__class__.__name__}')
        super().__init__(model=model, initial_hidden=wrapped_filter.hidden, initial_controls=wrapped_filter.controls)
        self.wrapped_filter = wrapped_filter
        self.trans_model = model

    def step(self,
             new_controls: MultivariateRandomDistribution,
             measurements: MultivariateRandomDistribution
             ) -> Tuple[MultivariateRandomDistribution, MultivariateRandomDistribution]:
        hidden_cov, output_cov = self.wrapped_filter.step(new_controls, measurements)

        return MultivariateGaussian(
            mean=self.model.transform_hidden_from_wrapped(hidden_cov.get_mean()),
            covariance=self.model.transform_covariance_from_wrapped(hidden_cov.get_covariance()),
        ), output_cov
