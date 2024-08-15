""" Framework for dual estimation of transient vector and A-SOH"""
from typing import Tuple, Optional
from typing_extensions import Self

import numpy as np
from scipy.linalg import block_diag

from moirae.models.base import InputQuantities, OutputQuantities, GeneralContainer, HealthVariable, CellModel
from .utils.model import JointCellModelInterface, convert_vals_model_to_filter
from moirae.estimators.online import OnlineEstimator
from .filters.base import BaseFilter
from .filters.distributions import MultivariateRandomDistribution, MultivariateGaussian
from .filters.kalman.unscented import UnscentedKalmanFilter as UKF


class DualEstimator(OnlineEstimator):
    """
    In dual estimation, the transient vector and A-SOH are estimated by separate filters. This framework generally
    avoids numerical errors related to the magnitude differences between values pertaining to transient quantities and
    to the A-SOH parameters. However, correlations between these two types of quantities are partially lost, and the
    framework is more involved.
    """

    def __init__(self, transient_filter: BaseFilter, asoh_filter: BaseFilter) -> None:
        pass
