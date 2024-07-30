""" Defines functionality for Unscented Kálmán Filter (UKF) """
from typing import Union, Literal, Optional, Tuple, Dict, Collection

import numpy as np
from scipy.linalg import block_diag

from moirae.estimators.online import OnlineEstimator, MultivariateRandomDistribution
from moirae.estimators.online.distributions import MultivariateGaussian, DeltaDistribution
from moirae.estimators.online.utils import ensure_positive_semi_definite
from moirae.models.base import CellModel, HealthVariable, GeneralContainer, InputQuantities


def calculate_gain_matrix(cov_xy: np.ndarray, cov_y: np.ndarray) -> np.ndarray:
    """
    Function to calculate Kálmán gain, defined as
    L = cov_xy * (cov_y^(-1))

    Args:
        cov_xy: covariance between x and y (hidden states and outputs, respectively)
        cov_y: variance of y (output)
    """
    return np.matmul(cov_xy, np.linalg.inv(cov_y))


class JointUnscentedKalmanFilter(OnlineEstimator):
    """An Unscented Kalman Filter that estimates both transient and ASOH parameters.

    Args:
        model: Model used to describe the underlying physics of the storage system
        initial_asoh: Initial estimates for the health parameters of the battery, those being estimated or not
        initial_transients: Initial estimates for the transient states of the battery
        initial_inputs: Initial inputs to the system
        initial_covariance: The covariance matrix between all transient and SOH parameters.
            Assumes a variance of 1 and no covariance by default.
        updatable_asoh: Whether to estimate values for all updatable parameters (``True``),
            none of the updatable parameters (``False``),
            or only a select set of them (provide a list of names).
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
        normalize_asoh: Whether to normalize the ASOH terms to near 1 for the state used by
            the filter. If true, ``initial_covariance`` will be updated for you.
    """

    def __init__(self,
                 model: CellModel,
                 initial_asoh: HealthVariable,
                 initial_transients: GeneralContainer,
                 initial_inputs: InputQuantities,
                 initial_covariance: Optional[np.ndarray] = None,
                 alpha_param: float = 1.,
                 kappa_param: Union[float, Literal['automatic']] = 0.,
                 beta_param: float = 2.,
                 covariance_process_noise: Optional[np.ndarray] = None,
                 covariance_sensor_noise: Optional[np.ndarray] = None,
                 normalize_asoh: bool = False,
                 updatable_asoh: Union[bool, Collection[str]] = True):
        super().__init__(model, initial_asoh, initial_transients, initial_inputs, updatable_asoh)
        self.state = MultivariateGaussian(
            mean=np.concatenate([self._transients.to_numpy(), self._asoh.get_parameters()], axis=1)[0, :],
            covariance=np.zeros((self.num_hidden_dimensions,) * 2) if initial_covariance is None else initial_covariance
        )
        self.u = DeltaDistribution(mean=initial_inputs.to_numpy())

        # Determine any normalization factors
        self.normalize_asoh = normalize_asoh
        self.joint_normalization_factor = np.ones(self.num_hidden_dimensions)
        self.covariance_normalization = np.ones((self.num_hidden_dimensions, self.num_hidden_dimensions))
        if normalize_asoh:
            self.joint_normalization_factor[self.num_transients:] = self._asoh.get_parameters()
            # Special attention needs to be paid to cases whewre the initial provided value is 0.0. In these cases,
            # the normalization factor remains equal to 1. (variable is "un-normalized" and treated as raw.)
            self.joint_normalization_factor[self.joint_normalization_factor == 0] = 1.
            self.covariance_normalization = (self.joint_normalization_factor[None, :] *
                                             self.joint_normalization_factor[:, None])

            # Apply them
            self.state.mean /= self.joint_normalization_factor
            self.state.covariance /= self.covariance_normalization

        # Calculate augmented dimensions
        self._aug_len = int((2 * self.num_hidden_dimensions) + self.num_output_dimensions)

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
                f'Kappa parameter ({kappa_param:f}) must be > - Augmented_length L ({self._aug_len:d})!'
            self.kappa_param = kappa_param
        self.gamma_param = alpha_param * np.sqrt(self._aug_len + kappa_param)
        self.lambda_param = (alpha_param * alpha_param *
                             (self._aug_len + kappa_param)) - self._aug_len

        # Taking care of covariances
        if covariance_process_noise is None:  # assume std = 1.0e-8
            covariance_process_noise = 1.0e-08 * np.eye(self.num_hidden_dimensions)
        if covariance_sensor_noise is None:  # assume std = 1.0e-8
            covariance_sensor_noise = 1.0e-08 * np.eye(self.num_output_dimensions)
        assert covariance_process_noise.shape[0] == covariance_process_noise.shape[1], \
            'Process noise covariance matrix must be square, but it has shape ' + \
            str(covariance_process_noise.shape) + '!'
        assert covariance_sensor_noise.shape[0] == covariance_sensor_noise.shape[1], \
            'Sensor covariance matrix must be square, but it has shape ' + \
            str(covariance_sensor_noise.shape) + '!'
        assert covariance_process_noise.shape[0] == self.num_hidden_dimensions, \
            'Process noise covariance shape does not match hidden states!'
        assert covariance_sensor_noise.shape[0] == self.num_output_dimensions, \
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

    def _normalize_hidden_array(self, hidden_array: np.ndarray) -> np.ndarray:
        if self.normalize_asoh:
            return hidden_array / self.joint_normalization_factor
        return hidden_array

    def _denormalize_hidden_array(self, hidden_array: np.ndarray) -> np.ndarray:
        if self.normalize_asoh:
            hidden_array = hidden_array * self.joint_normalization_factor
        else:
            hidden_array = hidden_array.copy()  # So that the edits later don't affect it

        # Ensure that the ASOH parameters are nonnegative
        # TODO(vventuri): this is another terrible hotfix here, since some ECM parameters may need to be negative!
        hidden_array[:, self.num_transients:] = np.clip(hidden_array[:, self.num_transients:], 1e-16, np.inf)
        return hidden_array

    def step(self,
             u: MultivariateRandomDistribution,
             y: MultivariateRandomDistribution
             ) -> Tuple[MultivariateGaussian, MultivariateGaussian]:

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
        # Step 1: perform estimation update
        x_k_minus, y_k, cov_xy = self.estimation_update(sigma_pts=sigma_pts, u=u)
        # Step 2: correction step, adjust hidden states based on new measurement
        self.correction_update(x_k_minus=x_k_minus, y_hat=y_k, cov_xy=cov_xy, y=y)
        # Don't forget to update the internal control!
        self.u = u.model_copy(deep=True)
        # Return
        return y_k.model_copy(deep=True), self.state.model_copy(deep=True)

    def build_sigma_points(self) -> np.ndarray:
        """
        Function to build Sigma points.

        Returns:
            2D numpy array, where each row represents an "augmented state" consisting of hidden state, process noise,
            and sensor noise, in that order.
        """
        # Building augmented state (recall noise terms are all zero-mean!)
        x_aug = np.hstack((self.state.mean, np.zeros(self.num_hidden_dimensions + self.num_output_dimensions)))

        # Now, build the augmented covariance
        cov_aug = block_diag(self.state.covariance, self.cov_w, self.cov_v)
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
        dim = self.num_hidden_dimensions
        x_hid = sigma_pts[:, :dim].copy()
        w_hid = sigma_pts[:, dim:(2 * dim)].copy()
        v_hid = sigma_pts[:, (2 * dim):].copy()
        return x_hid, w_hid, v_hid

    def _evolve_hidden(self, hidden_states: np.ndarray, new_control: MultivariateRandomDistribution) -> np.ndarray:
        """
        Function used to evolve the hidden states obtained from the Sigma points

        Args:
            hidden_states: array of hidden states from breaking of the Sigma points
            new_control: new control variables to be given to the model

        Returns:
            - x_update: updated hidden states
        """
        # Get old control
        u_old = self.u.model_copy(deep=True)

        # Update hidden states
        x_update = self.update_hidden_states(hidden_states=hidden_states,
                                             previous_controls=u_old,
                                             new_controls=new_control)
        return x_update

    def _assemble_unscented_estimate(self, samples: np.ndarray) -> Dict:
        """
        Function that takes a collection of samples (either updated hidden states (evolved from the Sigma points) and
        assembles them into a new hidden state corresponding to x_k_minus (includes mean and covariance)

        Args:
            samples: array of evolved hidden states

        Returns:
            x_k_minus: new estimate of the hidden state (includes mean and covariance!)
        """
        # Start with mean
        mu = np.average(samples, axis=0, weights=self.mean_weights)

        # For the covariance, since there can be negative weights, we need to calculate things by hand
        diffs = samples - mu
        cov = self._get_unscented_covariance(array0=diffs)
        return {'mean': mu, 'covariance': cov}

    def _get_unscented_covariance(self, array0: np.ndarray, array1: np.ndarray = None) -> np.ndarray:
        """
        Functions that computes the unscented covariance between zero-mean arrays. If second array is not provided,
        this is equivalent to computing the unscented variance of the only provided array.
        """
        if array1 is None:
            array1 = array0
        cov = np.matmul(array0.T, np.matmul(np.diag(self.cov_weights), array1))
        return cov

    def estimation_update(self,
                          sigma_pts: np.ndarray,
                          u: MultivariateRandomDistribution) \
            -> Tuple[MultivariateGaussian, MultivariateGaussian, np.ndarray]:
        """
        Function to perform the estimation update from the Sigma points

        Args:
            sigma_pts: numpy array corresponding to the Sigma points built
            u: new controls to be used to evolve Sigma points

        Returns:
            - x_k_minus: new estimate of the hidden state corresponding to x_k_minus (includes mean and covariance!)
            - y_k: estimate of the output measurement (includes mean and covariance!)
            - cov_xy: covariance matrix between hidden state and output
        """
        # Step 1a: break up Sigma points to get hidden states, process errors, and sensor errors
        x_hid, w_hid, v_hid = self._break_sigma_pts(sigma_pts=sigma_pts)

        # Step 1b: evolve hidden states based on the model and the previous input
        x_updated = self._evolve_hidden(hidden_states=x_hid, new_control=u)
        # Don't forget to include process noise!
        x_updated += w_hid
        # Assemble x_k_minus
        x_k_minus_info = self._assemble_unscented_estimate(samples=x_updated)
        x_k_minus = MultivariateGaussian.model_validate(x_k_minus_info)

        # Step 1c: use updated hidden states to predict outputs
        y_preds = self.predict_measurement(x_updated, controls=u)
        # Don't forget to include sensor noise!
        y_preds += v_hid
        # Assemble y_hat
        y_k_info = self._assemble_unscented_estimate(samples=y_preds)
        y_k = MultivariateGaussian.model_validate(y_k_info)

        # Calculate covariance between hidden state and output
        cov_xy = self._get_unscented_covariance(array0=(x_updated - x_k_minus.mean), array1=(y_preds - y_k.mean))

        return x_k_minus, y_k, cov_xy

    def correction_update(self,
                          x_k_minus: MultivariateGaussian,
                          y_hat: MultivariateGaussian,
                          cov_xy: np.ndarray,
                          y: MultivariateRandomDistribution) -> None:
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
        innovation = y.get_mean() - y_hat.mean
        innovation = innovation.flatten()

        # Step 2c: update the hidden state mean and covariance
        x_k_hat_plus = x_k_minus.mean + np.matmul(l_k, innovation)
        cov_x_k_plus = x_k_minus.covariance - np.matmul(l_k, np.matmul(y_hat.covariance, l_k.T))

        # Make sure this new covariance is positive semi-definite
        cov_x_k_plus = ensure_positive_semi_definite(cov_x_k_plus)

        # Set internal state
        self.state.mean = x_k_hat_plus
        self.state.covariance = cov_x_k_plus


class UnscentedKalmanFilter(JointUnscentedKalmanFilter):
    """A Kalman filter which only operates on the transient states

    Args:
        model: Model used to describe the underlying physics of the storage system
        initial_asoh: Initial estimates for the health parameters of the battery, those being estimated or not
        initial_transients: Initial estimates for the transient states of the battery
        initial_inputs: Initial inputs to the system
        initial_covariance: The covariance matrix between all transient and SOH parameters.
            Assumes a variance of 1 and no covariance by default.
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
                 model: CellModel,
                 initial_asoh: HealthVariable,
                 initial_transients: GeneralContainer,
                 initial_inputs: InputQuantities,
                 initial_covariance: Optional[np.ndarray] = None,
                 alpha_param: float = 1.,
                 kappa_param: Union[float, Literal['automatic']] = 0.,
                 beta_param: float = 2.,
                 covariance_process_noise: Optional[np.ndarray] = None,
                 covariance_sensor_noise: Optional[np.ndarray] = None,
                 ):
        super().__init__(
            model,
            initial_asoh,
            initial_transients,
            initial_inputs,
            initial_covariance,
            alpha_param,
            kappa_param,
            beta_param,
            covariance_process_noise,
            covariance_sensor_noise,
            updatable_asoh=False)
