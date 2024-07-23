"""Interfaces for running common workflows with Auto-SOH"""
import pandas as pd
from batdata.data import BatteryDataset

from asoh.estimators.online import OnlineEstimator
from asoh.models.base import CellModel, HealthVariable, GeneralContainer


def run_online_estimate(
        dataset: BatteryDataset,
        model: CellModel,
        initial_asoh: HealthVariable,
        initial_transients: GeneralContainer,
        estimator: OnlineEstimator
) -> pd.DataFrame:
    """Run an online estimation of battery parameters given a fixed dataset for the

    Args:
        dataset: Dataset containing the time series of a battery's performance
        model: Model describing the
        initial_asoh: Initial estimates for the state of health of the battery
        initial_transients: Initial estimates for the transient state of the batter
        estimator: Technique used to estimate the state of health
    Returns:
        Estimates of the parameters at all timesteps from the input dataset
    """
    raise NotImplementedError()
