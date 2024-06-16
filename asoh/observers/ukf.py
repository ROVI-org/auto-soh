"""Unscented Kálmán Filter (UKF) - a state estimator which makes no assumptions about the model form of a dynamic system."""
from functools import cached_property

import numpy as np
from scipy.linalg import block_diag

from .base import BaseEstimator
from ..models.base import ControlState, Outputs, HealthModel, InstanceState


class UnscentedKalmanFilter(BaseEstimator):
    """Unscented Kalman Filter (UKF)


    Its initialization inputs are:
        hidden_states ---> initial hidden states as np.array
        covariance_hidden_states ---> initial covariance of hidden states
                                        as np.array
        measurement_length ---> length of the measurement array
        input_length ---> length of the input array
        hidden_to_hidden_model ---> model used to update hidden states
                                    (takes in hidden state, input, and
                                    process error, as well as *argv and **kwargs)
        hidden_to_measure_model ---> model used to predict measurements from
                                    current hidden state (take in hidden
                                    state, input, sensor error, as well as *argv
                                    and **kwargs)
        alpha_param ---> tuning parameter 0.01 <= alpha <= 1 used to control the
                            spread of the sigma points; lower values keep sigma
                            points closer to the mean, alpha=1 effectively brings
                            the KF closer to Central Difference KF (default = 1.)
        kappa_param ---> tuning parameter  kappa >= 3 - aug_len; choose values
                        of kappa >=0 for positive semidefiniteness. (default = 0.)
        beta_param ---> tuning parameter beta >=0 used to incorporate knowledge
                        of prior distribution; for Gaussian use beta = 2
                        (default = 2.)
        h_param ---> DEPRECATED!! main tuning paramenter for CDKF (default = sqrt(3))
        covariance_process_noise ---> covariance of process noise as np.array
                                        (default = identity)
        covariance_sensor_noise ---> covariance of sensor noise as
                                     np.array(default = identity)
        keep_full_history ---> whether or not to keep full history
                                (default = True)
    """

    Cov_w: np.ndarray = ...
    """Noise associated with updating the state of a dynamic system"""
    Cov_v: np.ndarray = ...
    """Noise associated with measuring the outputs of the system"""

    cov_XY: np.ndarray = ...
    """Covariance between each hidden state and each output"""
    cov_Y: np.ndarray = ...
    """Covariance between each output"""

    U: np.ndarray | None = ...
    """Covariance of the previous timestep"""

    def __init__(self,
                 model: HealthModel,
                 state: InstanceState,
                 alpha_param: float = 1.,
                 kappa_param: float = 0.,
                 beta_param: float = 2.,
                 covariance_process_noise=None,
                 covariance_sensor_noise=None,
                 keep_full_history=True):
        super().__init__(model, state)

        # General inits
        self.Cov_Y = np.zeros((self.Y_len, self.Y_len))
        self.Cov_XY = np.zeros((self.X_len, self.Y_len))  # cov_hidden_output
        self.U = None
        self.L_k = np.zeros((self.X_len, self.Y_len))  # gain matrix

        # Augmeted dimensions
        # recall aug_len = hidden + noise_hidden + noise_measurement
        self.aug_len = self.X_len + self.X_len + self.Y_len

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
            covariance_process_noise = np.eye(self.X_len)
        if covariance_sensor_noise is None:  # assume std=1
            covariance_sensor_noise = np.eye(self.Y_len)
        assert covariance_process_noise.shape[0] == covariance_process_noise.shape[1], 'Covariance matrices must be square!'
        assert covariance_sensor_noise.shape[0] == covariance_sensor_noise.shape[1], 'Covariance matrices must be square!'
        assert covariance_process_noise.shape[0] == self.X_len, \
            'Process noise covariance shape does not match hidden states!'
        assert covariance_sensor_noise.shape[0] == self.Y_len, \
            'Sensor noise covariance shape does not match measurement lenght!'
        self.Cov_w = covariance_process_noise.copy()
        self.Cov_v = covariance_sensor_noise.copy()

        # Let's also take care of the weights for updating augmented state:
        mean_weights = 0.5 * np.ones((2 * self.aug_len + 1)) / (alpha_param * alpha_param * (self.aug_len + kappa_param))
        mean_weights[0] = self.lambda_param / (alpha_param * alpha_param * (self.aug_len + kappa_param))
        self.mean_weights = mean_weights.copy()
        cov_weights = mean_weights.copy()
        cov_weights[0] += 1 - (alpha_param * alpha_param) + beta_param
        self.cov_weights = cov_weights.copy()

        # Now, check if we want to keep the full history
        self.keep_history = keep_full_history
        if keep_full_history:
            self.X_minus_history = [self.X.copy()]
            self.X_plus_history = [self.X.copy()]
            self.Cov_X_minus_history = [self.Cov_X.copy()]
            self.Cov_X_plus_history = [self.Cov_X.copy()]
            self.Y_history = []
            self.Cov_Y_history = []
            self.Cov_XY_history = []
            self.y_err_hist = []
            # self.U_history = [] # i shouldn't need this

    @property
    def X(self) -> np.ndarray:
        """Current state of the system"""
        return self.state.full_state

    @cached_property
    def X_len(self) -> int:
        """Number of inputs"""
        return self.state.full_state.size

    @cached_property
    def Y_len(self):
        """Number of observable outputs"""
        return self.model.num_outputs

    def old_step(self, new_input, new_measure, *args, **kwargs):
        # Step 0: Create Sigma points of augmented state
        self._build_sigma_pts()
        # Step 1: perform update on the estimates of hidden state
        self._estimation_update(new_input, *args, **kwargs)
        # Step 2: apply necessary corrections based on the measurement
        y_err = self._correction_update(new_measure)
        return self.X, self.Cov_X, y_err

    def step(self, u: ControlState, y: Outputs, t_step: float):
        # Step 0: Create Sigma points of augmented state
        sigma_pts = self._build_sigma_pts()
        # Step 1: perform update on the estimates of hidden state
        self._estimation_update(sigma_pts, u.to_numpy(), t_step)
        # Step 2: apply necessary corrections based on the measurement
        y_err = self._correction_update(new_measure)
        return self.X, self.Cov_X, y_err

    def _build_sigma_pts(self) -> np.ndarray:
        """Generate Sigma points, which are points where the UKF will to predict updates in states

        Each row of the Sigma matrix includes points defining the hidden state,
        noise of the hidden state, and noise in the measurements.
        """
        # X_aug is basically the expected hidden state with the errors, which
        # are supposed to be 0 mean
        X_aug = np.hstack(
            (self.X, np.zeros((self.X_len + self.Y_len,)))
        )

        # Covariance matrix is block diagonal of hidden + errors
        cov_aug = block_diag(self.Cov_X, self.Cov_w, self.Cov_v)
        cov_aug = self._ensure_pos_semi_def(cov_aug)

        # Sigma points are the augmented "mean" + each individual row of the
        # transpose of the Cholesky decomposition of cov_aug, with a weighing
        # factor of plus and minus h_param
        sqrt_Cov_aug = np.linalg.cholesky(cov_aug).T
        aux_Sigma_Pts = np.vstack(
            (np.zeros((self.aug_len,)),
             self.gamma_param * sqrt_Cov_aug,
             -self.gamma_param * sqrt_Cov_aug)
        )

        # np.tile basically repeats the provided array as many times as given by
        # second argument tuple, in this case, into aug_len rows
        Sigma_pts = X_aug + aux_Sigma_Pts
        assert Sigma_pts.shape == (2 * self.aug_len + 1, self.aug_len), 'Dimensions of Sigma Points are incorrect!'
        return Sigma_pts

    def _estimation_update(self, Sigma_pts, control: ControlState, new_input, t_step):
        """

        Args:
            Sigma_pts: Points at which to test update to state, produced by :meth:`_build_Sigma_points`
            new_input: Control inputs at the next timestep
            t_step: Amount of time elapsed until the next timestep. Units: s

        Returns:
            -# x_hat_k_minus, Cov_x_k_minus, y_hat_k, Cov_y_k, Cov_xy_k_minus
        """
        # Separate the portions of the sigma points dealing with state, process noise, sensor noise
        x_hidden = Sigma_pts[:, :self.X_len].copy()
        w_hidden = Sigma_pts[:, self.X_len:(2 * self.X_len)].copy()
        v_hidden = Sigma_pts[:, (2 * self.X_len):].copy()
        u = self.U.copy()  # get old input, as it will update at the end of this

        # Let the model update the hidden state
        updated_xs = []
        for new_x in x_hidden:
            self.state.set_full_state(new_x)
            self.model.update(self.state, control, t_step)
            updated_xs.append(self.state.full_state)
        updated_xs = np.concatenate(updated_xs, axis=0)

        # TODO (wardlt): Stopped here

        # NOTE: these are the new hidden states, not the new augmented states!
        x_hat_k_minus = np.average(updated_xs, axis=0, weights=self.mean_weights)
        # x_hat_k_minus = np.matmul(self.mean_weights, updated_xs.T)
        assert x_hat_k_minus.shape == (self.X_len,), \
            'Wrong x_k_minus dimensions!'

        # Now, update the covariance
        # NOTE: we cannot use numpy.cov with argument aweights because some of
        #       the weights are negative!!!
        '''
        # CentralDiff Way
        Cov_x_k_minus = np.cov(updated_xs, 
                               rowvar=False, # our variables are columns, and 
                                             # rows are "observations"/samples 
                               aweights=self.cov_weights, # use weights
                               bias=True # divide by number of samples
                               # could have also used ddof = 0 to override bias
                               )
        '''
        # This is such that we could, if needed, implement
        # Unscented KF instead
        x_diffs = updated_xs - x_hat_k_minus
        Cov_x_k_minus = np.matmul(x_diffs.T,
                                  np.matmul(np.diag(self.cov_weights), x_diffs))
        assert Cov_x_k_minus.shape == (self.X_len, self.X_len), \
            'Wrong dimensions of Cov_x_k_minus!'

        # Now, let's update the output/measure estimate!
        # NOTE: Ideally, this should NOT use any *args NOR **kwargs, and should
        #       only depend on hidden state, input, sensor error
        # NOTE 2: This is no longer the case, as we need to provide Delta_t in
        #         order to perform denoising on some hidden states
        updated_ys = self.h2m_model(updated_xs, new_input, v_hidden, *args, **kwargs)
        # A minor problem is that, if self.Y_len = 1, np.average colapses the
        # result into a dimensionless np.array; but, if we use keepdims=True,
        # then y_hat_k.shape will be (1, self.Y_len)
        # Therefore, it seems we have to treat these cases differently
        y_hat_k = np.average(updated_ys, axis=0, weights=self.mean_weights)  # ,
        #  keepdims=self.Y_len==1)
        assert y_hat_k.shape == (self.Y_len,), 'Wrong y_hat dimensions!'

        # Covariance of y (again, aweights throws an error)
        '''
        # CentralDiff way
        Cov_y_k = np.cov(updated_ys, 
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
        Cov_y_k = np.matmul(y_diffs.T,
                            np.matmul(np.diag(self.cov_weights),
                                      y_diffs)
                            )
        assert Cov_y_k.shape == (self.Y_len, self.Y_len), \
            'Wrong Cov_Y_k dimensions!'
        # Finally, compute covariance X-to-Y!
        Cov_xy_k_minus = np.matmul(x_diffs.T,
                                   np.matmul(np.diag(self.cov_weights),
                                             y_diffs))
        # TODO: Do I need to ensure positive semi-definiteness here?
        assert Cov_xy_k_minus.shape == (self.X_len, self.Y_len), \
            'Wrong Cov_xy_k dimensions!'

        # Storing new estimates
        self.X = x_hat_k_minus.copy()
        self.Cov_X = Cov_x_k_minus.copy()
        self.Y = y_hat_k.copy()
        self.Cov_Y = Cov_y_k.copy()
        self.Cov_XY = Cov_xy_k_minus.copy()
        self.U = new_input.copy()

        # adding to history
        if self.keep_history:
            self.X_minus_history.append(x_hat_k_minus.copy())
            self.Cov_X_minus_history.append(Cov_x_k_minus.copy())
            self.Y_history.append(y_hat_k.copy())
            self.Cov_Y_history.append(Cov_y_k.copy())
            self.Cov_XY_history.append(Cov_xy_k_minus.copy())

    # Function to perform necessary corrections:
    # L_k_gain, x_hat_k_plus, Cov_x_k_plus
    def _calculate_gain_matrix(self):  # set self.L_k gain matrix
        L_k = np.matmul(self.Cov_XY, np.linalg.inv(self.Cov_Y))
        self.L_k = L_k.copy()

    def _correction_update(self, new_measure):
        # get gain
        self._calculate_gain_matrix()
        # estimate new state
        assert new_measure.shape == (self.Y_len,), 'Wrong shape for measure!'
        L_k = self.L_k.copy()
        y_err = new_measure - self.Y.copy()
        x_hat_k_plus = self.X.copy() + np.matmul(L_k, y_err)
        # Theory:
        Cov_x_k_plus = self.Cov_X - np.matmul(L_k, np.matmul(self.Cov_Y, L_k.T))
        # TODO: is there an equivalent Joseph-form updated (used for helping
        #       with positive definiteness) for Sigma Point KF?
        Cov_x_k_plus = self._ensure_pos_semi_def(Cov_x_k_plus)

        # Updating internal variables
        self.X = x_hat_k_plus.copy()
        self.Cov_X = Cov_x_k_plus

        # adding to history
        if self.keep_history:
            self.X_plus_history.append(x_hat_k_plus.copy())
            self.Cov_X_plus_history.append(Cov_x_k_plus.copy())
            self.y_err_hist.append(y_err.copy())

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
        '''
        Ref.: 1Nicholas J. Higham, “Computing a Nearest Symmetric Positive
        Semidefinite Matrix,” Linear Algebra and its Applications, 103,
        103–118, 1988
        '''
        # Perform singular value decomposition
        _, S_diagonal, V_complex_conjugate = np.linalg.svd(Sig)
        H_matrix = np.matmul(V_complex_conjugate.T, \
                             np.matmul(S_diagonal, V_complex_conjugate))
        return (Sig + Sig.T + H_matrix + H_matrix.T) / 4
