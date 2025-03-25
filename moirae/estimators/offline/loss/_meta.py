"""Loss functions which combine other loss functions"""
import numpy as np

from ._base import BaseLoss


class AdditiveLoss(BaseLoss):
    """Loss function which combines multiple loss functions

    Supply a list of loss functions and weights for each.

    Args:
        losses: List of loss functions, defined as pairs of (weight, loss) values
    """

    def __init__(self, losses: list[tuple[float, BaseLoss]]):
        if len(losses) == 0:
            raise ValueError('At least 1 loss function required')
        _l = losses[0][1]
        super().__init__(
            cell_model=_l.cell_model,
            transient_state=_l.state,
            asoh=_l.asoh,
            observations=_l.observations
        )
        self._losses = list(losses)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        output = np.zeros((x.shape[0],))
        for weight, loss in self._losses:
            output += weight * loss(x)
        return output
