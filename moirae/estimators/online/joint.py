""" Framework for joint estimation of transient vector and A-SOH"""
from typing import Tuple, Self, Optional

import numpy as np
from scipy.linalg import block_diag

from moirae.models.base import InputQuantities, OutputQuantities, GeneralContainer, HealthVariable, CellModel
from utils.model import JointCellModelInterface, convert_vals_model_to_filter
from moirae.estimators.online import HealthEstimator
from filters.base import BaseFilter
from filters.distributions import MultivariateRandomDistribution, MultivariateGaussian
from filters.kalman.unscented import UnscentedKalmanFilter as UKF


class JointEstimator(HealthEstimator):
    """
    In joint estimation, the transient vector and A-SOH values are estimated jointly, in a single array, using a single
    filter. Because of this simplicity, all that it truly needs to operate is the filter object
    """
    joint_model: JointCellModelInterface

    def __init__(self, filter: BaseFilter):
        self.filter = filter
        self.joint_model = filter.model

    # TODO (vventuri): convert this later to allow for uncertain input quantities
    def _convert_inputs(self, inputs: InputQuantities) -> MultivariateRandomDistribution:
        return convert_vals_model_to_filter(inputs)

    # TODO (vventuri): convert this later to allow for uncertain measurement quantities
    def _convert_measurements(self, measurements: OutputQuantities) -> MultivariateRandomDistribution:
        return convert_vals_model_to_filter(measurements)

    def step(self,
             inputs: InputQuantities,
             measurements: OutputQuantities) -> Tuple[GeneralContainer, HealthVariable, OutputQuantities]:
        """
        Main step functionality of the joint estimator.

        Args:
            inputs: new inputs
            measurements: measured quantities from the cell

        Returns:
            estimated_transient: estimated transient vector
            estimated_asoh: estimated A-SOH
            predicted_output: predicted values of the output quantities
        """

        refactored_inputs = self._convert_inputs(inputs=inputs)
        refactored_measurements = self._convert_measurements(measurements=measurements)

        joint_estimate, output_predicted = self.filter.step(new_controls=refactored_inputs,
                                                            measurements=refactored_measurements)

        estimated_transient, estimated_asoh = \
            self.joint_model.create_cell_model_inputs(hidden_states=np.atleast_2d(joint_estimate.get_mean()))

        predicted_outputs = measurements.model_copy(deep=True)
        predicted_outputs.from_numpy(output_predicted.get_mean())

        return estimated_transient, estimated_asoh, predicted_outputs

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
        """
        # Start by assembling the joint model wrapper
        joint_model = JointCellModelInterface(cell_model=cell_model,
                                              asoh=initial_asoh,
                                              transients=initial_transients,
                                              input_template=initial_inputs)

        # Prepare objects to be given to UKF
        # Joint hidden state
        joint_hidden_mean = joint_model.create_hidden_state(transients=initial_transients, asoh=initial_asoh)
        if covariance_joint is None:
            joint_hidden_covariance = block_diag(covariance_transient, covariance_asoh)
        else:
            joint_hidden_covariance = covariance_joint
        joint_initial_hidden = MultivariateGaussian(mean=joint_hidden_mean, covariance=joint_hidden_covariance)
        # Initial controls
        initial_controls = convert_vals_model_to_filter(model_quantities=initial_inputs,
                                                        uncertainty_matrix=inputs_uncertainty)

        # Initialize filter
        ukf = UKF(model=joint_model,
                  initial_hidden=joint_initial_hidden,
                  initial_controls=initial_controls)
        '''
        ukf = UKF(model = joint_model,
                  initial_hidden: MultivariateGaussian,
                  initial_controls: MultivariateRandomDistribution,
                  alpha_param: float = 1.,
                  kappa_param: Union[float, Literal['automatic']] = 0.,
                  beta_param: float = 2.,
                  covariance_process_noise: Optional[np.ndarray] = None,
                  covariance_sensor_noise: Optional[np.ndarray] = None
                  )
        '''

        return JointEstimator(filter=ukf)
