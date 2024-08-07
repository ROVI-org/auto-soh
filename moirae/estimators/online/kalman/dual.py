"""Dual UKF implementation"""
import numpy as np
from typing import Tuple, Optional, Union, Literal, Collection

from moirae.estimators.online import MultivariateRandomDistribution, OnlineEstimator
from moirae.estimators.online.distributions import MultivariateGaussian
from moirae.estimators.online.kalman.unscented import JointUnscentedKalmanFilter, build_sigma_points
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

        # Update the covariance for the ASOH
        theta_k_minus = self._ukf_asoh.state.get_mean()
        cov_theta_k_minus = self._ukf_asoh.state.get_covariance() + self._ukf_asoh.cov_w

        # Compute the updated transient states
        self._ukf_transient.asoh.update_parameters(theta_k_minus, self._updatable_names)  # Ensure operating on the same mean
        transients_sigma = self._ukf_transient.build_sigma_points()
        x_k_minus, y_k, cov_xy = self._ukf_transient.estimation_update(sigma_pts=transients_sigma, u=u)

        # Generate updated transient states and estimate outputs
        #  using Sigma points from the ASOH UKF
        self._ukf_asoh.transients.from_numpy(self._x_km1_plus)
        theta_sigma = self._ukf_asoh.build_sigma_points()
        _, theta_w_hid, theta_v_hid = self._ukf_asoh._break_sigma_pts(sigma_pts=theta_sigma)

        theta_sigma_with_transient = np.concatenate([
            np.repeat(self._x_km1_plus[None, :], theta_sigma.shape[0], axis=0),
            theta_sigma[:, :self._ukf_asoh.num_hidden_dimensions]
        ], axis=1)
        tswt_updated = self._update_hidden_states(theta_sigma_with_transient, self.u, u)
        tswt_updated[:, self.num_transients:] += theta_w_hid
        tswt_yhat = self._predict_measurement(tswt_updated, u)
        tswt_yhat += theta_v_hid

        # Store the x_km1_plus
        self._x_km1_plus = ...



