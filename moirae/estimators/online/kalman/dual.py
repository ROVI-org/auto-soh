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
        self._asoh_state = MultivariateGaussian(
            mean=initial_asoh.get_parameters(self._updatable_names),
            covariance=initial_asoh_covariance,
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
            updatable_asoh=updatable_asoh,
            normalize_asoh=normalize_asoh
        )

        self._x_km1_plus = initial_transients.to_numpy()

    def step(self,
             u: MultivariateRandomDistribution,
             y: MultivariateRandomDistribution) \
        -> Tuple[MultivariateRandomDistribution, MultivariateRandomDistribution]:

        # Update the covariance for the ASOH
        theta_k_minus = self._ukf_asoh.state.get_mean()
        cov_theta_k_minus = self._ukf_asoh.state.get_covariance() + self._ukf_asoh.cov_w

        # Compute the updated transient states
        self._ukf_transient._asoh.set_value(theta_k_minus, self._updatable_names)  # Ensure operating on the same mean
        sigma_pts = self._ukf_transient.build_sigma_points()
        x_k_minus, y_k, cov_xy = self._ukf_transient.estimation_update(sigma_pts=sigma_pts, u=u)

        # Make sigma points for the ASOH UK
        theta_sigma = build_sigma_points(
            self._asoh_state,
            self._ukf_asoh.cov_w,
            self._ukf_asoh.cov_v,
            self._ukf_asoh.gamma_param
        )
        theta_sigma_with_transient = np.concatenate([
            np.repeat(self._x_km1_plus, theta_sigma.shape[0], axis=0),
            theta_sigma
        ])
        # TODO (wardlt): Just call the CellModel directly
        tswt_updated = self.update_hidden_states(theta_sigma_with_transient, self.u, u)
        tswt_yhat = self.predict_measurement(tswt_updated, u)




