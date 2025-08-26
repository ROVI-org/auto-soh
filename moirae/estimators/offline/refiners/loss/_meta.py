"""Loss functions which combine other loss functions"""
from dataclasses import dataclass

import numpy as np

from battdat.data import BatteryDataset

from ._base import BaseLoss


@dataclass
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
            transient_state=_l.transient_state,
            asoh=_l.asoh,
        )
        self._losses = list(losses)

    def __call__(self, x: np.ndarray, observations: BatteryDataset) -> np.ndarray:
        output = np.zeros((x.shape[0],))
        for weight, loss in self._losses:
            output += weight * loss(x, observations)
        return output
