from typing import Union, List

from pytest import fixture
import numpy as np


@fixture
def coarse_soc():
    return np.linspace(0, 1, 10)


@fixture
def fine_soc():
    return np.linspace(0, 1, 100)


@fixture()
def const_ocv():
    def const_ocv(soc_vals: Union[float, np.ndarray]) -> float:
        """
        Returns constant OCV value
        """
        return 3.5

    return const_ocv


def linear_ocv(soc_vals: Union[float, np.ndarray]) -> Union[float, List[float]]:
    """
    Returns OCV value that is linearly dependent of SOC provided
    """
    volts = np.array(3 + soc_vals)
    volts = volts.astype(float)
    return volts.tolist()
