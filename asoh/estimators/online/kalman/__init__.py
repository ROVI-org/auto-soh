""" Colection of classes and functions pertinent to Kálmán filters """
from asoh.estimators.online import HiddenState, OutputMeasurements
from asoh.estimators.online.base import MultivariateGaussian


class KalmanHiddenState(MultivariateGaussian, HiddenState):
    """
    Hidden state to be used by Kálmán filters
    """
    pass


class KalmanOutputMeasurement(MultivariateGaussian, OutputMeasurements):
    """
    Ouput measurements for Kálmán filters
    """
    pass
