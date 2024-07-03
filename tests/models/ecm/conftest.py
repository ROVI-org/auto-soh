# General imports
from typing import Union, List
import numpy as np

coarse_soc = np.linspace(0, 1, 10)
fine_soc = np.linspace(0, 1, 100)


def const_ocv(soc_vals: Union[float, np.ndarray]) -> float:
    return 3.5


def linear_ocv(soc_vals: Union[float, np.ndarray]) -> Union[float, List[float]]:
    volts = 3 + soc_vals
    volts = volts.astype(float)
    return volts.tolist()


def realistic_ocv(
        soc_vals: Union[float, np.ndarray]) -> Union[float, List[float]]:
    x_scale = 0.9
    x_off = 0.05
    y_scale = 0.1
    y_off = 3.5
    mod_soc = x_scale * soc_vals
    mod_soc += x_off
    volts = np.log(mod_soc / (1 - mod_soc))
    volts *= y_scale
    volts += y_off
    volts = volts.astype(float)
    return volts.tolist()
