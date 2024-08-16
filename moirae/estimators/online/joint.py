""" Framework for joint estimation of transient vector and A-SOH"""
from typing import Tuple, Optional
from typing_extensions import Self

import numpy as np
from scipy.linalg import block_diag

from moirae.models.base import InputQuantities, OutputQuantities, GeneralContainer, HealthVariable, CellModel
from .utils.model import JointCellModelWrapper, convert_vals_model_to_filter
from moirae.estimators.online import OnlineEstimator
from .filters.base import BaseFilter
from .filters.distributions import MultivariateRandomDistribution, MultivariateGaussian
from .filters.kalman.unscented import UnscentedKalmanFilter as UKF


class JointEstimator(OnlineEstimator):
    """
    Estimate the transient vector and A-SOH values are estimated jointly, in a single array, using a single filter.

    Create a joint estimator by supplying a single filter.

    Args:
        joint_filter: A filter configured to operate using a :class:`~moirae.models.base.CellModel`.
    """

    def __init__(self, joint_filter: BaseFilter):
        if not isinstance(joint_filter.model, JointCellModelWrapper):
            raise ValueError('The joint estimator only works for a filter which uses a JointCellModel to describe the '
                             'physics')
        model_interface = joint_filter.model
        super().__init__(
            cell_model=model_interface.cell_model,
            initial_asoh=model_interface.asoh,
            initial_transients=model_interface.transients,
            initial_inputs=model_interface.inputs,
            updatable_asoh=model_interface.asoh_inputs
        )
        self.filter = joint_filter
        self.joint_model = joint_filter.model

    @property
    def state(self) -> MultivariateRandomDistribution:
        return self.filter.hidden.model_copy(deep=True)

    def get_estimated_state(self) -> Tuple[GeneralContainer, HealthVariable]:
        joint_state = self.state
        estimated_asoh, estimated_transient = self.joint_model.create_cell_model_inputs(hidden_states=joint_state)
        return estimated_transient, estimated_asoh

    def step(self, inputs: InputQuantities, measurements: OutputQuantities) -> \
            Tuple[MultivariateRandomDistribution, MultivariateRandomDistribution]:
        # TODO (vventuri): convert this later to allow for uncertain input quantities
        refactored_inputs = convert_vals_model_to_filter(inputs)
        refactored_measurements = convert_vals_model_to_filter(measurements)

        joint_estimate, output_predicted = self.filter.step(new_controls=refactored_inputs,
                                                            measurements=refactored_measurements)

        return joint_estimate, output_predicted

    @classmethod
    def initialize_unscented_kalman_filter(cls,
                                           cell_model: CellModel,
                                           # TODO (vventuri): add degrataion_model as an option here
                                           initial_asoh: HealthVariable,
                                           initial_transients: GeneralContainer,
                                           initial_inputs: InputQuantities,
                                           covariance_joint: Optional[np.ndarray] = None,
                                           covariance_transient: Optional[np.ndarray] = None,
                                           covariance_asoh: Optional[np.ndarray] = None,
                                           inputs_uncertainty: Optional[np.ndarray] = None,
                                           joint_covariance_process_noise: Optional[np.ndarray] = None,
                                           transient_covariance_process_noise: Optional[np.ndarray] = None,
                                           asoh_covariance_process_noise: Optional[np.ndarray] = None,
                                           covariance_sensor_noise: Optional[np.ndarray] = None,
                                           **filter_args) -> Self:
        """
        Function to help the user initialize a UKF-based joint estimation without needing to define the filter and
        model wrapper.

        Args:
            cell_model: CellModel to be used
            degratation_model: DegradationModel to be used; if None is passed, assume A-SOH degration estimated solely
                                from data
            initial_asoh: initial A-SOH
            initial_transients: initial transient vector
            initial_inputs: initial input quantities
            covariance_joint: specifies the raw (un-normalized) covariance of the joint state; it is the preferred
                method of assembling the initial joint state
            covariance_transient: specifies the raw (un-normalized) covariance of the transient state; it is not used if
                covariance_joint was provided
            covariance_asoh: specifies the raw (un-normalized) covariance of the A-SOH; it is not used if
                covariance_joint was provided
            inputs_uncertainty: uncertainty matrix of the inputs; if not provided, assumes inputs are exact
            joint_covariance_process_noise: process noise covariance for the joint state, considering transient and
                A-SOH noises
            transient_covariance_process_noise: process noise for transient update; only used if
                joint_covariance_process_noise was not provided
            asoh_covariance_process_noise: process noise for A-SOH update; only used if joint_covariance_process_noise
                was not provided
            covariance_sensor_noise: sensor noise for outputs; if denoising is applied, must match the proper
                dimensionality
            **filter_args: additional keywords to be given the the UKF; can include alpha_param, beta_param, and
                kappa_param
        """
        # Start by assembling the joint model wrapper
        joint_model = JointCellModelWrapper(cell_model=cell_model,
                                            asoh=initial_asoh,
                                            transients=initial_transients,
                                            inputs=initial_inputs)

        # Prepare objects to be given to UKF
        # Joint hidden state
        joint_hidden_mean = joint_model.create_hidden_state(transients=initial_transients, asoh=initial_asoh)
        if covariance_joint is None:
            joint_hidden_covariance = block_diag(covariance_transient, covariance_asoh)
        else:
            joint_hidden_covariance = covariance_joint
        joint_initial_hidden = MultivariateGaussian(mean=joint_hidden_mean[0, :], covariance=joint_hidden_covariance)
        # Initial controls
        initial_controls = convert_vals_model_to_filter(model_quantities=initial_inputs,
                                                        uncertainty_matrix=inputs_uncertainty)

        # Process noise
        if joint_covariance_process_noise is None:
            if (transient_covariance_process_noise is not None) and (asoh_covariance_process_noise is not None):
                joint_covariance_process_noise = block_diag(transient_covariance_process_noise,
                                                            asoh_covariance_process_noise)

        # Initialize filter
        ukf = UKF(model=joint_model,
                  initial_hidden=joint_initial_hidden,
                  initial_controls=initial_controls,
                  covariance_process_noise=joint_covariance_process_noise,
                  covariance_sensor_noise=covariance_sensor_noise,
                  **filter_args)

        return JointEstimator(joint_filter=ukf)
