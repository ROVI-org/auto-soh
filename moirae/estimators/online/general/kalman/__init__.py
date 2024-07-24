""" Colection of classes and functions pertinent to Kálmán filters """
from moirae.estimators.online import HiddenState, OutputMeasurements
from moirae.estimators.online.general import MultivariateGaussian


class KalmanHiddenState(MultivariateGaussian, HiddenState):
    """
    Hidden state to be used by Kálmán filters
    """
    pass


class KalmanOutputMeasurement(MultivariateGaussian, OutputMeasurements):
    """
    Output measurements for Kálmán filters
    """
    pass
