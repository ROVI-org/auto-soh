""" Defines functionality for Unscented Kálmán Filter (UKF) """
from typing import Union, Literal, Optional, Tuple

import numpy as np

from asoh.estimators.online import ModelFilterInterface, OnlineEstimator, ControlVariables, OutputMeasurements
from asoh.estimators.online.base.kalman import KalmanHiddenState, KalmanOutputMeasurement


class UnscentedKalmanFilter(OnlineEstimator):
    """
    Class that defines the basic functionality of the Unscented Kalman Filter

    Args:
        model: model describing the system
        initial_state: initial hidden state of the system
        alpha_param: tuning parameter 0.001 <= alpha <= 1 used to control the
                    spread of the sigma points; lower values keep sigma
                    points closer to the mean, alpha=1 effectively brings
                    the KF closer to Central Difference KF (default = 1.)
        kappa_param: tuning parameter  kappa >= 3 - aug_len; choose values
                     of kappa >=0 for positive semidefiniteness. (default = 0.)
        beta_param: tuning parameter beta >=0 used to incorporate knowledge
                        of prior distribution; for Gaussian use beta = 2
                        (default = 2.)
        covariance_process_noise: covariance of process noise (default = 1.0e-8 * identity)
        covariance_sensor_noise: covariance of sensor noise as (default = 1.0e-8 * identity)
    """

    def __init__(self,
                 model: ModelFilterInterface,
                 initial_state: KalmanHiddenState,
                 alpha_param: float = 1.,
                 kappa_param: Union[float, Literal['automatic']] = 0.,
                 beta_param: float = 2.,
                 covariance_process_noise: Optional[np.ndarray] = None,
                 covariance_sensor_noise: Optional[np.ndarray] = None):
        self.model = model
        self.state = initial_state.model_copy(deep=True)
        assert model.num_hidden_dimensions == initial_state.num_dimensions, \
            'Model expects %d hidden dimensions, but initial state has %d!' % \
            (model.num_hidden_dimensions, initial_state.num_dimensions)

        # Calculate augmented dimensions
        self._aug_len = int((2 * model.num_hidden_dimensions) + model.num_output_dimensions)

        # Tuning parameters check and save
        assert alpha_param >= 0.001, 'Alpha parameter should be >= 0.001!'
        assert alpha_param <= 1, 'Alpha parameter must be <= 1!'
        assert beta_param >= 0, 'Beta parameter must be >= 0!'
        self.alpha_param = alpha_param
        self.beta_param = beta_param
        if kappa_param == 'automatic':
            self.kappa_param = 3 - self._aug_len
        else:
            assert self._aug_len + kappa_param > 0, \
                'Kappa parameter (%f) must be > - Augmented_length L (%d)!' % (kappa_param, self._aug_len)
            self.kappa_param = kappa_param
        self.gamma_param = alpha_param * np.sqrt(self._aug_len + kappa_param)
        self.lambda_param = (alpha_param * alpha_param *
                             (self._aug_len + kappa_param)) - self._aug_len

        # Taking care of covariances
        if covariance_process_noise is None:  # assume std = 1.0e-4
            covariance_process_noise = 1.0e-08 * np.eye(self.model.num_hidden_dimensions)
        if covariance_sensor_noise is None:  # assume std = 1.0e-4
            covariance_sensor_noise = 1.0e-08 * np.eye(self.model.num_output_dimensions)
        assert covariance_process_noise.shape[0] == covariance_process_noise.shape[1], \
            'Process noise covariance matrix must be square, but it has shape ' + \
            str(covariance_process_noise.shape) + '!'
        assert covariance_sensor_noise.shape[0] == covariance_sensor_noise.shape[1], \
            'Sensor covariance matrix must be square, but it has shape ' + \
            str(covariance_sensor_noise.shape) + '!'
        assert covariance_process_noise.shape[0] == self.model.num_hidden_dimensions, \
            'Process noise covariance shape does not match hidden states!'
        assert covariance_sensor_noise.shape[0] == self.model.num_output_dimensions, \
            'Sensor noise covariance shape does not match measurement length!'
        self.cov_w = covariance_process_noise.copy()
        self.cov_v = covariance_sensor_noise.copy()

        # Finally, we can set the weights for the mean and covariance updates
        mean_weights = 0.5 * np.ones((2 * self._aug_len + 1))
        mean_weights[0] = self.lambda_param
        mean_weights /= (alpha_param * alpha_param * (self._aug_len + kappa_param))
        self.mean_weights = mean_weights.copy()
        cov_weights = mean_weights.copy()
        cov_weights[0] += 1 - (alpha_param * alpha_param) + beta_param
        self.cov_weights = cov_weights.copy()

    def step(self,
             u: ControlVariables,
             y: OutputMeasurements
             ) -> Tuple[KalmanOutputMeasurement, KalmanHiddenState]:
        """
        Steps the UKF

        Args:
            u: new control variables
            y: new measurements

        Returns:
            Predicted output based on model evolution, as well as corrected hidden state
        """
        # Step 0: build Sigma points
        sigma_pts = self.build_sigma_points()
        # Step 1a: perform estimation step (evolution of the Sigma points)
        evolved_sigma_pts, x_k_minus = self.estimation_update(sigma_pts=sigma_pts, u=u)
        # Step 1b: predict output measurements from x_hat_k_minus
        y_hat, cov_xy = self.predict_outputs(evolved_sigma_pts)
        # Step 2: correction step, adjust hidden states based on new measurement
        self.correction_update(y_hat=y_hat, y=y, cov_xy=cov_xy)
        # Return
        return (y_hat.model_copy(deep=True), self.state.model_copy(deep=True))

    def build_sigma_points(self) -> np.ndarray:
        """
        Function to build Sigma points
        """
        pass

    def estimation_update(self,
                          sigma_pts: np.ndarray,
                          u: ControlVariables) -> Tuple[np.ndarray, KalmanHiddenState]:
        """
        Function to perform the estimation update from the Sigma points

        Args:
            sigma_pts: numpy array corresponding to the Sigma points built
            u: new controls to be used to evolve Sigma points

        Returns:
            Evolved Sigma points, as well as new estimate of the hidden state corresponding to x_k_minus
        """
        pass

    def predict_outputs(self,
                        evolved_sigma_pts: np.ndarray) -> Tuple[KalmanOutputMeasurement, np.ndarray]:
        """
        Function to predict outputs from evolved Sigma points.

        Args:
            evolved_sigma_pts: numpy array of the updated Sigma points

        Returns:
            Tuple composed of y_hat, the predicted output (containing mean and covariance), as well as cov_xy, the
            updated covariance matrix between hidden states and outputs
        """
        pass

    def correction_update(self,
                          y_hat: KalmanOutputMeasurement,
                          y: OutputMeasurements,
                          cov_xy: np.ndarray) -> None:
        pass
