""" Framework for dual estimation of transient vector and A-SOH"""
from typing import Tuple, Optional, TypedDict
from typing_extensions import Self, NotRequired

import numpy as np

from moirae.models.base import InputQuantities, OutputQuantities, GeneralContainer, HealthVariable, CellModel
from .model.cell import convert_vals_model_to_filter
from .model.cell import CellModelWrapper, DegradationModelWrapper
from moirae.estimators.online import OnlineEstimator
from .filters.base import BaseFilter
from .filters.distributions import MultivariateRandomDistribution, MultivariateGaussian
from .filters.kalman.unscented import UnscentedKalmanFilter as UKF
from .filters.kalman.unscented import UKFTuningParameters


class DualUKFTuningParameters(TypedDict):
    """
    Auxiliary class to help provide tuning parameters to each filter in the dual estimation framework defined by
    ~:class:`~moirae.estimators.online.dual.DualEstimator`

    Args:
        transient: tuning parameters for the transient filter
        asoh: tuning parameters for the A-SOH filter
    """
    transient: NotRequired[UKFTuningParameters]
    asoh: NotRequired[UKFTuningParameters]

    @classmethod
    def defaults(cls) -> Self:
        return {'transient': UKFTuningParameters.defaults(), 'asoh': UKFTuningParameters.defaults()}


class DualEstimator(OnlineEstimator):
    """
    In dual estimation, the transient vector and A-SOH are estimated by separate filters. This framework generally
    avoids numerical errors related to the magnitude differences between values pertaining to transient quantities and
    to the A-SOH parameters. However, correlations between these two types of quantities are partially lost, and the
    framework is more involved.
    """

    def __init__(self, transient_filter: BaseFilter, asoh_filter: BaseFilter) -> None:
        if not isinstance(transient_filter.model, CellModelWrapper):
            raise ValueError('The dual estimator only works with a filter which uses a CellModelWrapper to describe the'
                             ' dynamics of the transient states')
        if not isinstance(asoh_filter.model, DegradationModelWrapper):
            raise ValueError('The dual estimator only works with a filter which uses a DegradationModelWrapper to '
                             'describe the degradation of the A-SOH!')

        # Storing necessary objects
        cell_wrapper = transient_filter.model
        asoh_wrapper = asoh_filter.model
        super().__init__(
            cell_model=cell_wrapper.cell_model,
            initial_asoh=asoh_wrapper.asoh,
            initial_transients=cell_wrapper.transients,
            initial_inputs=cell_wrapper.inputs,
            updatable_asoh=asoh_wrapper.asoh_inputs
        )
        self.trans_filter = transient_filter
        self.asoh_filter = asoh_filter
        self.cell_wrapper = cell_wrapper
        self.asoh_wrapper = asoh_wrapper

    @property
    def state(self) -> MultivariateRandomDistribution:
        # We need to get the hidden states on both filters
        transient_hidden = self.trans_filter.hidden.model_copy(deep=True)
        asoh_hidden = self.asoh_filter.hidden.model_copy(deep=True)
        return transient_hidden.combine_with(asoh_hidden)

    def get_estimated_state(self) -> Tuple[GeneralContainer, HealthVariable]:
        # We need to get the hidden states on both filters
        transient_hidden = self.trans_filter.hidden.model_copy(deep=True)
        asoh_hidden = self.asoh_filter.hidden.model_copy(deep=True)
        # Convert
        estimated_transient = self.transients.make_copy(values=transient_hidden.get_mean())
        estimated_asoh = self.asoh_wrapper._convert_hidden_to_asoh(hidden_states=asoh_hidden.get_mean())
        return estimated_transient, estimated_asoh

    def step(self, inputs: InputQuantities, measurements: OutputQuantities) -> \
            Tuple[MultivariateRandomDistribution, MultivariateRandomDistribution]:
        # Refactor inputs and measurements to filter-lingo
        # TODO (vventuri): convert this later to allow for uncertain input quantities
        refactored_inputs = convert_vals_model_to_filter(inputs)
        refactored_measurements = convert_vals_model_to_filter(measurements)

        # Collect posteriors from previous states to communicate between wrappers
        previous_trans_posterior, previous_asoh_posterior = self.get_estimated_state()

        # Give these to the wrappers
        self.cell_wrapper.asoh = previous_asoh_posterior.model_copy(deep=True)
        self.asoh_wrapper.transients = previous_trans_posterior.model_copy(deep=True)

        # Now, we can safely step each filter on its own
        transient_estimate, output_pred_trans = self.trans_filter.step(new_controls=refactored_inputs,
                                                                       measurements=refactored_measurements)
        asoh_estimate, output_pred_asoh = self.asoh_filter.step(new_controls=refactored_inputs,
                                                                measurements=refactored_measurements)

        # TODO (vventuri): how to we adequately combine the output predictions from both filters?
        return transient_estimate.combine_with((asoh_estimate,)), output_pred_trans

    @classmethod
    def initialize_unscented_kalman_filter(
        cls,
        cell_model: CellModel,
        # TODO (vventuri): add degrataion_model as an option here
        initial_asoh: HealthVariable,
        initial_transients: GeneralContainer,
        initial_inputs: InputQuantities,
        covariance_transient: np.ndarray,
        covariance_asoh: np.ndarray,
        inputs_uncertainty: Optional[np.ndarray] = None,
        transient_covariance_process_noise: Optional[np.ndarray] = None,
        asoh_covariance_process_noise: Optional[np.ndarray] = None,
        covariance_sensor_noise: Optional[np.ndarray] = None,
        filter_args: Optional[DualUKFTuningParameters] = DualUKFTuningParameters.defaults()
    ) -> Self:
        """
        Function to help the user initialize a UKF-based dual estimation without needing to define each filter and
        model wrapper individually.

        Args:
            cell_model: CellModel to be used
            degratation_model: DegradationModel to be used; if None is passed, assume A-SOH degration estimated solely
                                from data
            initial_asoh: initial A-SOH
            initial_transients: initial transient vector
            initial_inputs: initial input quantities
            covariance_transient: specifies the raw (un-normalized) covariance of the transient state; it is not used if
                covariance_joint was provided
            covariance_asoh: specifies the raw (un-normalized) covariance of the A-SOH; it is not used if
                covariance_joint was provided
            inputs_uncertainty: uncertainty matrix of the inputs; if not provided, assumes inputs are exact
            transient_covariance_process_noise: process noise for transient update; only used if
                joint_covariance_process_noise was not provided
            asoh_covariance_process_noise: process noise for A-SOH update; only used if joint_covariance_process_noise
                was not provided
            covariance_sensor_noise: sensor noise for outputs; if denoising is applied, must match the proper
                dimensionality
            filter_args: dictionary where keys must be either 'transient' or 'asoh', and values must be a dictionary in
                which keys are keyword arguments for the UKFs (such as `alpha_param`)
        """
        # Assemble the wrappers
        cell_wrapper = CellModelWrapper(cell_model=cell_model,
                                        asoh=initial_asoh,
                                        transients=initial_transients,
                                        inputs=initial_inputs)
        asoh_wrapper = DegradationModelWrapper(cell_model=cell_model,
                                               asoh=initial_asoh,
                                               transients=initial_transients,
                                               inputs=initial_inputs)

        # Prepare multivariate Gaussians
        transients_hidden = convert_vals_model_to_filter(model_quantities=initial_transients,
                                                         uncertainty_matrix=covariance_transient)
        asoh_hidden = MultivariateGaussian(mean=initial_asoh.get_parameters().flatten(),
                                           covariance=covariance_asoh)
        initial_controls = convert_vals_model_to_filter(model_quantities=initial_inputs)

        # Initialize filters
        trans_filter = UKF(model=cell_wrapper,
                           initial_hidden=transients_hidden,
                           initial_controls=initial_controls,
                           covariance_process_noise=transient_covariance_process_noise,
                           covariance_sensor_noise=covariance_sensor_noise,
                           **filter_args.get('transient', UKFTuningParameters.defaults()))
        asoh_filter = UKF(model=asoh_wrapper,
                          initial_hidden=asoh_hidden,
                          initial_controls=initial_controls,
                          covariance_process_noise=asoh_covariance_process_noise,
                          covariance_sensor_noise=covariance_sensor_noise,
                          **filter_args.get('asoh', UKFTuningParameters.defaults()))

        if filter_args is not None:
            if 'transient' in filter_args.keys():
                for tuning_param, value in filter_args['transient'].items():
                    setattr(trans_filter, tuning_param, value)
            if 'asoh' in filter_args.keys():
                for tuning_param, value in filter_args['asoh'].items():
                    setattr(trans_filter, tuning_param, value)

        return DualEstimator(transient_filter=trans_filter, asoh_filter=asoh_filter)
