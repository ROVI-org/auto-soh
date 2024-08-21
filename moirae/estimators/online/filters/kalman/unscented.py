""" Definition of Unscented Kálmán Filter (UKF)"""
from typing import Union, Literal, Optional, Tuple, Dict, TypedDict
from typing_extensions import NotRequired, Self
from functools import cached_property

import numpy as np
from scipy.linalg import block_diag

from moirae.estimators.online.filters.distributions import (MultivariateRandomDistribution,
                                                            MultivariateGaussian,
                                                            DeltaDistribution)
from moirae.estimators.online.filters.base import ModelWrapper, BaseFilter
from .utils import ensure_positive_semi_definite, calculate_gain_matrix


def assemble_unscented_estimate_from_samples(samples: np.ndarray,
                                             mean_weights: np.ndarray,
                                             cov_weights: np.ndarray) -> Dict:
    """
    Function that takes a collection of samples and computes the relative mean and covariance based on the weights
    provided

    Args:
        samples: array of evolved hidden states
        mean_weights: weights to be used by the computation of the mean
        cov_weights: weights to be used by the computation of the covariance

    Returns:
        Dictionary of containing 'mean' and 'covariance'
    """
    # Start with mean
    mu = np.average(samples, axis=0, weights=mean_weights)

    # For the covariance, since there can be negative weights, we need to calculate things by hand
    diffs = samples - mu
    cov = compute_unscented_covariance(cov_weights=cov_weights, array0=diffs)
    return {'mean': mu, 'covariance': cov}


def compute_unscented_covariance(cov_weights: np.ndarray,
                                 array0: np.ndarray,
                                 array1: Optional[np.ndarray] = None,
                                 ) -> np.ndarray:
    """
    Function that computes the unscented covariance between zero-mean arrays. If second array is not provided,
    this is equivalent to computing the unscented variance of the only provided array.
    """
    if array1 is None:
        array1 = array0
    cov = np.matmul(array0.T, np.matmul(np.diag(cov_weights), array1))
    return cov


class UKFTuningParameters(TypedDict):
    """
    Auxiliary class to help provide tuning parameters to
    ~:class:`~moirae.estimators.online.filters.kalman.UnscentedKalmanFilter`

    Args:
        alpha_param: alpha parameter to UKF
        beta_param: beta parameter to UKF
        kappa_param: kappa parameter to UKF
    """
    alpha_param: NotRequired[float]
    beta_param: NotRequired[float]
    kappa_param: NotRequired[Union[float, Literal['automatic']]]

    @classmethod
    def defaults(cls) -> Self:
        return {'alpha_param': 1., 'kappa_param': 0., 'beta_param': 2.}


class UnscentedKalmanFilter(BaseFilter):
    """
    Class that defines the functionality of the Unscented Kalman Filter

    Args:
        model: model describing the system
        initial_hidden: initial hidden state of the system
        initial_controls: initial control on the system
        alpha_param: tuning parameter 0.001 <= alpha <= 1 used to control the spread of the sigma points; lower values
            keep sigma points closer to the mean, alpha=1 effectively brings the KF closer to Central Difference KF
            (default = 1.)
        kappa_param: tuning parameter  kappa >= 3 - aug_len; choose values of kappa >=0 for positive semidefiniteness.
            (default = 0.)
        beta_param: tuning parameter beta >=0 used to incorporate knowledge of prior distribution; for Gaussian use
            beta = 2 (default = 2.)
        covariance_process_noise: covariance of process noise (default = 1.0e-8 * identity)
        covariance_sensor_noise: covariance of sensor noise as (default = 1.0e-8 * identity)
    """
    def __init__(self,
                 model: ModelWrapper,
                 initial_hidden: MultivariateGaussian,
                 initial_controls: MultivariateRandomDistribution,
                 covariance_process_noise: Optional[np.ndarray] = None,
                 covariance_sensor_noise: Optional[np.ndarray] = None,
                 alpha_param: float = 1.,
                 kappa_param: Union[float, Literal['automatic']] = 0.,
                 beta_param: float = 2.):
        # Store main parameters
        super().__init__(model=model, initial_hidden=initial_hidden, initial_controls=initial_controls)

        # Get number of hidden dimensions
        assert model.num_hidden_dimensions == initial_hidden.num_dimensions, \
            'Model expects %d hidden dimensions, but initial state has %d!' % \
            (model.num_hidden_dimensions, initial_hidden.num_dimensions)

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

        # Taking care of covariances
        if covariance_process_noise is None:  # assume std = 1.0e-8
            covariance_process_noise = 1.0e-08 * np.eye(self.model.num_hidden_dimensions)
        if covariance_sensor_noise is None:  # assume std = 1.0e-8
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

    @cached_property
    def gamma_param(self) -> float:
        return self.alpha_param * np.sqrt(self._aug_len + self.kappa_param)

    @cached_property
    def lambda_param(self) -> float:
        return (self.alpha_param * self.alpha_param * (self._aug_len + self.kappa_param)) - self._aug_len

    @cached_property
    def mean_weights(self) -> np.ndarray:
        mean_weights = 0.5 * np.ones((2 * self._aug_len + 1))
        mean_weights[0] = self.lambda_param
        mean_weights /= (self.alpha_param * self.alpha_param * (self._aug_len + self.kappa_param))
        return mean_weights

    @cached_property
    def cov_weights(self) -> np.ndarray:
        cov_weights = self.mean_weights.copy()
        cov_weights[0] += 1 - (self.alpha_param * self.alpha_param) + self.beta_param
        return cov_weights

    def step(self,
             new_controls: MultivariateRandomDistribution,
             measurements: DeltaDistribution,
             **kwargs
             ) -> Tuple[MultivariateRandomDistribution, MultivariateRandomDistribution]:
        """
        Steps the UKF

        Args:
            new_controls: new control variables
            measurements: new measurements
            **kwargs: keyword arguments to be given to the model

        Returns:
            Predicted output based on model evolution, as well as corrected hidden state
        """
        # Step 0: build Sigma points
        sigma_pts = self.build_sigma_points()
        # Step 1: perform estimation update
        x_k_minus, y_k, cov_xy = self.estimation_update(sigma_pts=sigma_pts, new_controls=new_controls, **kwargs)
        # Step 2: correction step, adjust hidden states based on new measurement
        self.correction_update(x_k_minus=x_k_minus, y_hat=y_k, cov_xy=cov_xy, y=measurements, **kwargs)
        # Don't forget to update the internal control!
        self.controls = new_controls.model_copy(deep=True)
        # Return
        return (self.hidden.model_copy(deep=True), y_k.model_copy(deep=True))

    def build_sigma_points(self) -> np.ndarray:
        """
        Function to build Sigma points.

        Returns:
            2D numpy array, where each row represents an "augmented state" consisting of hidden state, process noise,
            and sensor noise, in that order.
        """
        # Building augmented state (recall noise terms are all zero-mean!)
        x_aug = np.hstack((self.hidden.get_mean(),
                           np.zeros(self.model.num_hidden_dimensions + self.model.num_output_dimensions)))

        # Now, build the augmented covariance
        cov_aug = block_diag(self.hidden.get_covariance(), self.cov_w, self.cov_v)
        # Making sure this is positive semi-definite
        cov_aug = ensure_positive_semi_definite(cov_aug)

        # Sigma points are the augmented "mean" + each individual row of the transpose of the Cholesky decomposition of
        # Cov_aug, with a weighing factor of plus and minus gamma_param
        sqrt_cov_aug = np.linalg.cholesky(cov_aug).T
        aux_sigma_pts = np.vstack((np.zeros((self._aug_len,)),
                                   self.gamma_param * sqrt_cov_aug,
                                   -self.gamma_param * sqrt_cov_aug))
        sigma_pts = x_aug + aux_sigma_pts
        return sigma_pts

    def _break_sigma_pts(self, sigma_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Function to break Sigma points into its hidden state, process noise, and sensor noise parts

        Args:
            sigma_pts: Sigma points matrix to be broken up

        Returns:
            - x_hid: array corresponding to iterable of hidden states
            - w_hid: array corresponding to iterable of process noises
            - v_hid: array corresponding to iterable of sensor noises
        """
        dim = self.model.num_hidden_dimensions
        x_hid = sigma_pts[:, :dim].copy()
        w_hid = sigma_pts[:, dim:(2 * dim)].copy()
        v_hid = sigma_pts[:, (2 * dim):].copy()
        return x_hid, w_hid, v_hid

    def _evolve_hidden(self,
                       hidden_states: np.ndarray,
                       new_controls: MultivariateRandomDistribution,
                       **kwargs) -> np.ndarray:
        """
        Function used to evolve the hidden states obtained from the Sigma points

        Args:
            hidden_states: array of hidden states from breaking of the Sigma points
            new_controls: new control variables to be given to the model

        Returns:
            - x_update: updated hidden states
        """
        # Get old control
        u_old = self.controls.model_copy(deep=True)

        # Update hidden states
        x_update = self.model.update_hidden_states(hidden_states=hidden_states,
                                                   previous_controls=u_old.get_mean(),
                                                   new_controls=new_controls.get_mean(),
                                                   **kwargs)
        return x_update

    def _assemble_unscented_estimate(self, samples: np.ndarray) -> Dict:
        """
        Function that takes a collection of samples (either updated hidden states (evolved from the Sigma points) and
        assembles them into a new hidden state corresponding to x_k_minus (includes mean and covariance)

        Args:
            updated_hidden_states: array of evolved hidden states

        Returns:
            x_k_minus: new estimate of the hidden state (includes mean and covariance!)
        """
        unscented_info = assemble_unscented_estimate_from_samples(samples=samples,
                                                                  mean_weights=self.mean_weights,
                                                                  cov_weights=self.cov_weights)
        return unscented_info

    def _predict_outputs(self,
                         updated_hidden_states: np.ndarray,
                         controls: MultivariateRandomDistribution,
                         **kwargs) -> np.ndarray:
        """
        Function to predict outputs from evolved Sigma points.

        Args:
            updated_hidden_states: numpy array of the updated hidden states (includes process noise already)
            controls: control to be used for predicting outpus
            **kwargs: keyword arguments that are given to the model

        Returns:
            y_preds: predicted outputs based on provided hidden states and control
        """
        y_preds = self.model.predict_measurement(hidden_states=updated_hidden_states,
                                                 controls=controls.get_mean(), **kwargs)
        return y_preds

    def estimation_update(self,
                          sigma_pts: np.ndarray,
                          new_controls: MultivariateRandomDistribution,
                          **kwargs) -> Tuple[MultivariateGaussian, MultivariateGaussian, np.ndarray]:
        """
        Function to perform the estimation update from the Sigma points

        Args:
            sigma_pts: numpy array corresponding to the Sigma points built
            new_controls: new control to be used to evolve Sigma points
            **kwargs: keyword arguments that are given to the model

        Returns:
            - x_k_minus: new estimate of the hidden state corresponding to x_k_minus (includes mean and covariance!)
            - y_k: estimate of the output measurement (includes mean and covariance!)
            - cov_xy: covariance matrix between hidden state and output
        """
        # Step 1a: break up Sigma points to get hidden states, process errors, and sensor errors
        x_hid, w_hid, v_hid = self._break_sigma_pts(sigma_pts=sigma_pts)

        # Step 1b: evolve hidden states based on the model and the previous input
        x_updated = self._evolve_hidden(hidden_states=x_hid, new_controls=new_controls, **kwargs)
        # Don't forget to include process noise!
        x_updated += w_hid
        # Assemble x_k_minus
        x_k_minus_info = self._assemble_unscented_estimate(samples=x_updated)
        x_k_minus = MultivariateGaussian.model_validate(x_k_minus_info)

        # Step 1c: use updated hidden states to predict outpus
        y_preds = self._predict_outputs(updated_hidden_states=x_updated, controls=new_controls, **kwargs)
        # Don't forget to include sensor noise!
        y_preds += v_hid
        # Assemble y_hat
        y_k_info = self._assemble_unscented_estimate(samples=y_preds)
        y_k = MultivariateGaussian.model_validate(y_k_info)
        # Calculate covariance between hidden state and output
        cov_xy = compute_unscented_covariance(cov_weights=self.cov_weights,
                                              array0=(x_updated - x_k_minus.get_mean()),
                                              array1=(y_preds - y_k.get_mean()))

        return x_k_minus, y_k, cov_xy

    def correction_update(self,
                          x_k_minus: MultivariateGaussian,
                          y_hat: MultivariateGaussian,
                          cov_xy: np.ndarray,
                          y: DeltaDistribution) -> None:
        """
        Function to perform the correction update of the hidden state, based on the real measured output values.

        Args:
            x_k_minus: estimate of the hidden state P(x_k|y_(k-1))
            y_hat: output predictions P(y_k|y_k-1)
            cov_xy: covariance between hidden state and predicted output
            y: real measured output values
        """
        # Step 2a: calculate gain matrix
        l_k = calculate_gain_matrix(cov_xy=cov_xy, cov_y=y_hat.covariance)

        # Step 2b: compute Kálmán innovation (basically, the error in the output predictions)
        innovation = y.get_mean() - y_hat.get_mean()
        innovation = innovation.flatten()

        # Step 2c: update the hidden state mean and covariance
        x_k_hat_plus = x_k_minus.get_mean() + np.matmul(l_k, innovation)
        cov_x_k_plus = x_k_minus.get_covariance() - np.matmul(l_k, np.matmul(y_hat.get_covariance(), l_k.T))
        # Make sure this new covariance is positive semi-definite
        cov_x_k_plus = ensure_positive_semi_definite(cov_x_k_plus)
        # Set internal state
        self.hidden = MultivariateGaussian(mean=x_k_hat_plus, covariance=cov_x_k_plus)
