"""Unscented Kálmán Filter (UKF) - a state estimator which makes no assumptions about the model form of a dynamic system."""
import numpy as np
from scipy.linalg import block_diag

from .base import BaseEstimator
from ..models.base import ControlState, Outputs, HealthModel, InstanceState


class UnscentedKalmanFilter(BaseEstimator):
    """Unscented Kalman Filter (UKF)

    Args:
        model: Model describing system dynamics
        state: Initial state estimate
        alpha_param: tuning parameter 0.01 <= alpha <= 1 used to control the
                    spread of the sigma points; lower values keep sigma
                    points closer to the mean, alpha=1 effectively brings
                    the KF closer to Central Difference KF (default = 1.)
        kappa_param: tuning parameter  kappa >= 3 - aug_len; choose values
                     of kappa >=0 for positive semidefiniteness. (default = 0.)
        beta_param: tuning parameter beta >=0 used to incorporate knowledge
                        of prior distribution; for Gaussian use beta = 2
                        (default = 2.)
        covariance_process_noise: covariance of process noise (default = identity)
        covariance_sensor_noise: covariance of sensor noise as (default = identity)
    """

    cov_w: np.ndarray = ...
    """Noise associated with updating the state of a dynamic system"""
    cov_v: np.ndarray = ...
    """Noise associated with measuring the outputs of the system"""

    cov_XY: np.ndarray = ...
    """Covariance between each hidden state and each output"""
    cov_Y: np.ndarray = ...
    """Covariance between each output"""

    u_old: ControlState | None = ...
    """Control signal of the previous timestep"""

    def __init__(self,
                 model: HealthModel,
                 state: InstanceState,
                 alpha_param: float = 1.,
                 kappa_param: float = 0.,
                 beta_param: float = 2.,
                 covariance_process_noise: np.ndarray = None,
                 covariance_sensor_noise: np.ndarray = None):
        super().__init__(model, state)

        # General inits
        self.cov_Y = np.zeros((self.model.num_outputs, self.model.num_outputs))
        self.u_old = None
        self.L_k = np.zeros((self.state.num_params, self.model.num_outputs))  # gain matrix

        # Augmented dimensions
        # recall aug_len = hidden + noise_hidden + noise_measurement
        self.aug_len = self.state.num_params + self.state.num_params + self.model.num_outputs

        # Tuning parameters
        assert alpha_param >= 0.01, 'Alpha parameter must be >= 0.01!'
        assert alpha_param <= 1, 'Alpha parameter must be <= 1!'
        assert beta_param >= 0, 'Beta parameter must be >= 0!'
        self.alpha_param = alpha_param
        self.beta_param = beta_param
        if kappa_param == 'auto':
            self.kappa_param = 3 - self.aug_len
        else:
            assert self.aug_len + kappa_param > 0, \
                'Kappa parameter must be > -L!'
            self.kappa_param = kappa_param
        self.gamma_param = alpha_param * np.sqrt(self.aug_len + kappa_param)
        self.lambda_param = (alpha_param * alpha_param *
                             (self.aug_len + kappa_param)) - self.aug_len

        # Taking care of covariances
        if covariance_process_noise is None:
            covariance_process_noise = np.eye(self.state.num_params)
        if covariance_sensor_noise is None:  # assume std=1
            covariance_sensor_noise = np.eye(self.model.num_outputs)
        assert covariance_process_noise.shape[0] == covariance_process_noise.shape[1], 'Covariance matrices must be square!'
        assert covariance_sensor_noise.shape[0] == covariance_sensor_noise.shape[1], 'Covariance matrices must be square!'
        assert covariance_process_noise.shape[0] == self.state.num_params, \
            'Process noise covariance shape does not match hidden states!'
        assert covariance_sensor_noise.shape[0] == self.model.num_outputs, \
            'Sensor noise covariance shape does not match measurement lenght!'
        self.cov_w = covariance_process_noise.copy()
        self.cov_v = covariance_sensor_noise.copy()

        # Let's also take care of the weights for updating augmented state:
        mean_weights = 0.5 * np.ones((2 * self.aug_len + 1)) / (alpha_param * alpha_param * (self.aug_len + kappa_param))
        mean_weights[0] = self.lambda_param / (alpha_param * alpha_param * (self.aug_len + kappa_param))
        self.mean_weights = mean_weights.copy()
        cov_weights = mean_weights.copy()
        cov_weights[0] += 1 - (alpha_param * alpha_param) + beta_param
        self.cov_weights = cov_weights.copy()

    def step(self, u: ControlState, y: Outputs, t_step: float):
        # Step 0: Create Sigma points of augmented state
        sigma_pts = self._build_sigma_pts()
        # Step 1: perform update on the estimates of hidden state
        cov_xy_k_minus, y_hat_k, cov_y_k = self._estimation_update(sigma_pts, u, t_step)
        # Step 2: apply necessary corrections based on the measurement
        self._correction_update(y, y_hat_k, cov_xy_k_minus, cov_y_k)
        # TODO (wardlt): Have this return the diagnostics which Victor was tracking

    def _build_sigma_pts(self) -> np.ndarray:
        """Generate Sigma points, which are points where the UKF will to predict updates in states

        Each row of the Sigma matrix includes points defining the hidden state,
        noise of the hidden state, and noise in the measurements.
        """
        # x_aug is basically the expected hidden state with the errors, which
        # are supposed to be 0 mean
        x_aug = np.hstack(
            (self.state.full_state, np.zeros((self.state.num_params + self.model.num_outputs,)))
        )

        # Covariance matrix is block diagonal of hidden + errors
        cov_aug = block_diag(self.state.covariance, self.cov_w, self.cov_v)
        cov_aug = self._ensure_pos_semi_def(cov_aug)

        # Sigma points are the augmented "mean" + each individual row of the
        # transpose of the Cholesky decomposition of cov_aug, with a weighing
        # factor of plus and minus h_param
        sqrt_cov_aug = np.linalg.cholesky(cov_aug).T
        aux_sigma_pts = np.vstack(
            (np.zeros((self.aug_len,)),
             self.gamma_param * sqrt_cov_aug,
             -self.gamma_param * sqrt_cov_aug)
        )

        # np.tile basically repeats the provided array as many times as given by
        # second argument tuple, in this case, into aug_len rows
        sigma_pts = x_aug + aux_sigma_pts
        assert sigma_pts.shape == (2 * self.aug_len + 1, self.aug_len), 'Dimensions of Sigma Points are incorrect!'
        return sigma_pts

    def _estimation_update(self, sigma_pts: np.ndarray, u_new: ControlState, t_step: float):
        """Update state estimation given the control signal and current state

        Updates the mean and covariance matrix of :attr:`state`

        Args:
            sigma_pts: Points at which to test update to state, produced by :meth:`_build_Sigma_points`
            u_new: Control signal measured at new timestep
            t_step: Amount of time elapsed until the next timestep. Units: s

        Returns:
           - cov_xy_k_minus: Estimated covariance matrix between state and outputs
           - y_hat_k: Estimated outputs
           - cov_y_k: Covariance between estimated outputs
        """

        # Special case: use current control if previous is not yet set
        if self.u_old is None:
            self.u_old = u_new

        # Separate the portions of the sigma points dealing with state, process noise, sensor noise
        x_hidden = sigma_pts[:, :self.state.num_params].copy()
        w_hidden = sigma_pts[:, self.state.num_params:(2 * self.state.num_params)].copy()
        v_hidden = sigma_pts[:, (2 * self.state.num_params):].copy()

        # Let the model update the hidden state
        updated_xs = []
        for new_x, new_w in zip(x_hidden, w_hidden):
            self.state.set_full_state(new_x)
            self.model.update(self.state, self.u_old, t_step)
            updated_xs.append(self.state.full_state + new_w)  # Apply the process noise
        updated_xs = np.array(updated_xs)

        # NOTE: these are the new hidden states, not the new augmented states!
        x_hat_k_minus = np.average(updated_xs, axis=0, weights=self.mean_weights)
        assert x_hat_k_minus.shape == (self.state.num_params,), 'Wrong x_k_minus dimensions!'

        # Now, update the covariance
        x_diffs = updated_xs - x_hat_k_minus
        cov_x_k_minus = np.matmul(x_diffs.T, np.matmul(np.diag(self.cov_weights), x_diffs))
        assert cov_x_k_minus.shape == (self.state.num_params, self.state.num_params), 'Wrong dimensions of cov_x_k_minus!'

        # Now, let's update the output/measure estimate!
        # NOTE: Ideally, this should NOT use any *args NOR **kwargs, and should
        #       only depend on hidden state, input, sensor error
        # NOTE 2: This is no longer the case, as we need to provide Delta_t in
        #         order to perform denoising on some hidden states
        updated_ys = []
        for updated_x, new_v in zip(updated_xs, v_hidden):
            self.state.set_full_state(updated_x)
            y_new = self.model.output(self.state, u_new)
            updated_ys.append(y_new.to_numpy() + new_v)  # Add in the sensor noise
        updated_ys = np.array(updated_ys)

        # A minor problem is that, if self.model.num_outputs = 1, np.average collapses the
        # result into a dimensionless np.array; but, if we use keepdims=True,
        # then y_hat_k.shape will be (1, self.model.num_outputs)
        # Therefore, it seems we have to treat these cases differently
        # TODO (wardlt): Would `np.atleast_2d` work?
        y_hat_k = np.average(updated_ys, axis=0, weights=self.mean_weights)
        assert y_hat_k.shape == (self.model.num_outputs,), 'Wrong y_hat dimensions!'

        # Covariance of y (again, aweights throws an error)
        '''
        # CentralDiff way
        cov_y_k = np.cov(updated_ys,
                         rowvar=False, # our variables are columns, and
                                        # rows are "observations"/samples
                         aweights=self.cov_weights, # use weights
                         bias=True # divide by number of samples
                         # could have also used ddof = 0 to override bias
                         )
        '''
        # Unscented KF way
        y_diffs = updated_ys - y_hat_k
        # np.tile(y_hat_k, ((2 * self.aug_len + 1), 1))
        cov_y_k = np.matmul(y_diffs.T,
                            np.matmul(np.diag(self.cov_weights),
                                      y_diffs)
                            )
        assert cov_y_k.shape == (self.model.num_outputs, self.model.num_outputs), 'Wrong Cov_Y_k dimensions!'
        # Finally, compute covariance X-to-Y!
        cov_xy_k_minus = np.matmul(x_diffs.T,
                                   np.matmul(np.diag(self.cov_weights),
                                             y_diffs))
        # TODO (vventuri): Do I need to ensure positive semi-definiteness here?
        assert cov_xy_k_minus.shape == (self.state.num_params, self.model.num_outputs), 'Wrong Cov_xy_k dimensions!'

        # Storing new estimates
        self.state.covariance = cov_x_k_minus
        self.u_old = u_new.copy(deep=True)

        return cov_xy_k_minus, y_hat_k, cov_y_k

    def _calculate_gain_matrix(self, cov_xy, cov_y):
        """Compute the gain matrix"""
        return np.matmul(cov_xy, np.linalg.inv(cov_y))

    def _correction_update(self, new_measure: Outputs, y_hat_k: np.ndarray, cov_xy, cov_y) -> np.ndarray:
        """Determine a correction to the state estimate, both mean and covariance

        Args:
            new_measure: Measurement to compare against correction
            y_hat_k: Estimated output given current state
            cov_xy: Covariance matrix between state and outputs, determined during :meth:`_estimation_update`
            cov_y: Covariance between output states, determined during :meth:`_estimation_update`
        Returns:
            Observed error between state estimation and covered error
        """
        # get gain
        l_k = self._calculate_gain_matrix(cov_xy, cov_y)

        # estimate new state
        y_err = new_measure.to_numpy() - y_hat_k
        x_hat_k_plus = self.state.full_state + np.matmul(l_k, y_err)

        # Update the estimate of the covariance
        cov_x_k_plus = self.state.covariance - np.matmul(l_k, np.matmul(cov_y, l_k.T))
        # TODO: is there an equivalent Joseph-form updated (used for helping
        #       with positive definiteness) for Sigma Point KF?
        cov_x_k_plus = self._ensure_pos_semi_def(cov_x_k_plus)

        # Updating internal variables
        self.state.set_full_state(x_hat_k_plus)
        self.state.covariance = cov_x_k_plus
        return y_err

    # Check for positive semi-definiteness
    def _ensure_pos_semi_def(self, Sig):
        if np.allclose(Sig, Sig.T):  # checking if it is symmetric
            try:
                # checking if positive semi-definiteness
                np.linalg.cholesky(Sig)  # throws LinAlgError if not
                return Sig.copy()
            except np.linalg.LinAlgError:
                return self._enforce_pos_semi_def(Sig).copy()
        else:
            return self._enforce_pos_semi_def(Sig).copy()

    # Enforce positive semi-definite covariances
    def _enforce_pos_semi_def(self, Sig):
        """
        Ref.: 1Nicholas J. Higham, “Computing a Nearest Symmetric Positive
        Semidefinite Matrix,” Linear Algebra and its Applications, 103,
        103–118, 1988
        """
        # Perform singular value decomposition
        _, S_diagonal, V_complex_conjugate = np.linalg.svd(Sig)
        H_matrix = np.matmul(V_complex_conjugate.T, np.matmul(S_diagonal, V_complex_conjugate))
        return (Sig + Sig.T + H_matrix + H_matrix.T) / 4
