"""Dual UKF implementation"""
from typing import Tuple, Optional, Union, Literal, Collection

import numpy as np
from scipy.linalg import block_diag

from moirae.estimators.online import MultivariateRandomDistribution, OnlineEstimator
from moirae.estimators.online.distributions import MultivariateGaussian
from moirae.estimators.online.kalman.unscented import JointUnscentedKalmanFilter
from moirae.models.base import GeneralContainer, HealthVariable, InputQuantities, CellModel


class DualUnscentedKalmanFilter(OnlineEstimator):

    def __init__(self,
                 model: CellModel,
                 initial_asoh: HealthVariable,
                 initial_transients: GeneralContainer,
                 initial_inputs: InputQuantities,
                 initial_transient_covariance: Optional[np.ndarray] = None,
                 initial_asoh_covariance: Optional[np.ndarray] = None,
                 alpha_transient_param: float = 1.,
                 kappa_transient_param: Union[float, Literal['automatic']] = 0.,
                 beta_transient_param: float = 2.,
                 alpha_asoh_param: float = 1.,
                 kappa_asoh_param: Union[float, Literal['automatic']] = 0.,
                 beta_asoh_param: float = 2.,
                 covariance_transient_process_noise: Optional[np.ndarray] = None,
                 covariance_asoh_process_noise: Optional[np.ndarray] = None,
                 covariance_sensor_noise: Optional[np.ndarray] = None,
                 normalize_asoh: bool = False,
                 updatable_asoh: Union[bool, Collection[str]] = True):
        super().__init__(
            model=model,
            initial_asoh=initial_asoh,
            initial_transients=initial_transients,
            initial_inputs=initial_inputs,
            updatable_asoh=updatable_asoh
        )
        self._ukf_transient = JointUnscentedKalmanFilter(
            model=model,
            initial_asoh=initial_asoh,
            initial_transients=initial_transients,
            initial_inputs=initial_inputs,
            initial_covariance=initial_transient_covariance,
            alpha_param=alpha_transient_param,
            beta_param=beta_transient_param,
            kappa_param=kappa_transient_param,
            covariance_process_noise=covariance_transient_process_noise,
            normalize_asoh=False,
            updatable_asoh=False
        )
        self._ukf_asoh = JointUnscentedKalmanFilter(
            model=model,
            initial_asoh=initial_asoh,
            initial_transients=initial_transients,
            initial_inputs=initial_inputs,
            initial_covariance=initial_asoh_covariance,
            alpha_param=alpha_asoh_param,
            beta_param=beta_asoh_param,
            kappa_param=kappa_asoh_param,
            covariance_process_noise=covariance_asoh_process_noise,
            covariance_sensor_noise=covariance_sensor_noise,
            updatable_transients=False,
            updatable_asoh=updatable_asoh,
            normalize_asoh=normalize_asoh
        )

        self._x_km1_plus = initial_transients.to_numpy()[0, :]

    def _step(self, u: MultivariateRandomDistribution, y: MultivariateRandomDistribution) \
            -> Tuple[MultivariateRandomDistribution, MultivariateRandomDistribution]:

        # Step -1: communicate transient posterior from previous iteration from one estimator to the other
        # Transients
        asoh_x_k_minus_mean = self._ukf_asoh.state.get_mean()
        self._ukf_transient.asoh.update_parameters(asoh_x_k_minus_mean, self._updatable_names)
        # A-SOH
        self._ukf_asoh.transients.from_numpy(self._x_km1_plus)

        # Step 0: build A-SOH sigma points in both estimators
        # Transients
        transients_sigma = self._ukf_transient.build_sigma_points()
        # A-SOH
        asoh_sigma = self._ukf_asoh.build_sigma_points()
        _, asoh_w_hid, asoh_v_hid = self._ukf_asoh._break_sigma_pts(sigma_pts=asoh_sigma)

        # Step 1: perform "estimation" steps
        # (for the A-SOH estimator hidden state, the mean stays the same, but process noise is added to the covariance)
        # Transients
        transients_x_k_minus, transients_y_hat, transients_cov_xy = \
            self._ukf_transient.estimation_update(sigma_pts=transients_sigma, u=u)
        # A-SOH: we need to be more careful, as we have to propagate the old transients through the sigma points, use
        #        the new transients for the output predictions, but then assure correct the dimensionality of the
        #        matrices to compute covariances correctly
        # TODO (vventuri): we are basically re-writing the estimation step, which is rather silly and wasteful.
        # Concatenate the old transient states with the A-SOH
        asoh_sigma_with_transient = np.concatenate([
            np.repeat(self._x_km1_plus[None, :], asoh_sigma.shape[0], axis=0),
            asoh_sigma[:, :self._ukf_asoh.num_hidden_dimensions]
        ], axis=1)
        asoh_with_transient_updated = self._update_hidden_states(asoh_sigma_with_transient, self.u, u)
        asoh_with_transient_updated[:, self.num_transients:] += asoh_w_hid  # adding process noise term to A-SOH
        asoh_ypred = self._predict_measurement(asoh_with_transient_updated, u)
        asoh_ypred += asoh_v_hid  # adding sensor noise to predictions
        # Assemble "estimate" and predictions
        asoh_x_k_minus = MultivariateGaussian(self._ukf_asoh._assemble_unscented_estimate(asoh_sigma))
        asoh_y_hat = MultivariateGaussian(self._ukf_asoh._assemble_unscented_estimate(asoh_ypred))
        # Now, compute covariance
        asoh_cov_xy = self._ukf_asoh._get_unscented_covariance(array0=(asoh_sigma - asoh_x_k_minus.get_mean()),
                                                               array1=(asoh_ypred - asoh_y_hat.get_mean()))

        # Step 2: perform correction steps
        # Transient
        self._ukf_transient.correction_update(x_k_minus=transients_x_k_minus,
                                              y_hat=transients_y_hat,
                                              cov_xy=transients_cov_xy)
        self._ukf_asoh.correction_update(x_k_minus=asoh_x_k_minus,
                                         y_hat=asoh_y_hat,
                                         cov_xy=asoh_cov_xy)

        # Store the x_km1_plus
        self._x_km1_plus = self._ukf_transient.state.model_copy(deep=True)

        # Now, prepare returns
        # Joint state
        # TODO (vventuri): I really dislike this hack... the first we do with this is split it into transient and A-SOH
        #                  in the public step function...
        transient_x_k_plus = self._ukf_transient.state
        asoh_x_k_plus = self._ukf_asoh.state
        joint_mean = np.hstack((transient_x_k_plus.get_mean(), asoh_x_k_plus.get_mean()))
        joint_cov = block_diag(transient_x_k_plus.get_covariance(), asoh_x_k_plus.get_covariance())
        joint_state = MultivariateGaussian(mean=joint_mean, covariance=joint_cov)
        # Predictions
        # TODO (vventuri) 1: this simple averaging will be problematic when denoising is introduced, and the outputs of
        #                   each filter end up having different dimensions.
        # TODO (vventur) 2: this also assumes that each prediction is independent of one another, which is obviously not
        #                   true. While, for now, this should not be an issue, it can lead us to be more confident in
        #                   the predictions than what we really should be...
        y_hat_mean = 0.5 * (transients_y_hat.get_mean() + asoh_y_hat.get_mean())
        y_hat_cov = 0.25 * (transients_y_hat.get_covariance() + asoh_y_hat.get_covariance())
        y_hat = MultivariateGaussian(mean=y_hat_mean, covariance=y_hat_cov)

        return (joint_state, y_hat)
