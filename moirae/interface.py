"""Interfaces for running common workflows with Auto-SOH"""
import pandas as pd
from batdata.data import BatteryDataset

from moirae.estimators.online import OnlineEstimator


def run_online_estimate(
        dataset: BatteryDataset,
        estimator: OnlineEstimator
) -> pd.DataFrame:
    """Run an online estimation of battery parameters given a fixed dataset for the

    Args:
        dataset: Dataset containing the time series of a battery's performance
        estimator: Technique used to estimate the state of health, which is built using
            a physics model which describes the cell and initial guesses for the battery
            transient and health states.
    Returns:
        Estimates of the parameters at all timesteps from the input dataset
    """
    raise NotImplementedError()
