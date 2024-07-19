""" Defines basic functionality for Joint UKF Estimator """
from typing import Union, Tuple
from abc import abstractmethod

import numpy as np
from scipy.linalg import block_diag

from asoh.models.base import AdvancedStateOfHealth, TransientVector, InputQuantities, OutputQuantities
from asoh.estimators.online import ControlVariables
from asoh.estimators.online.joint import ModelJointEstimatorInterface, JointOnlineEstimator
from asoh.estimators.online.general.kalman import KalmanHiddenState, KalmanOutputMeasurement
from asoh.estimators.online.general.kalman.unscented import UnscentedKalmanFilter as UKF


class ModelJointUKFInterface(ModelJointEstimatorInterface):
    """
    Class to interface model with Joint UKF.

    Args:
        asoh: initial Advanced State of Health (A-SOH) of the system
        transient: initial transiden hidden state of the syste
        control: initial control to the system
        normalize_joint_state: determines if the joint state should be made up of the raw values provided, or normalized
    """
    def __init__(self,
                 asoh: AdvancedStateOfHealth,
                 transient: TransientVector,
                 control: InputQuantities,
                 normalize_joint_state: bool = False) -> None:
        # Initialize parent class
        super().__init__(asoh=asoh, transient=transient, control=control, normalize_joint_state=normalize_joint_state)
        # Because everything in the UKF formalism is a Gaussian variable, we need to make sure we know how to scale the
        # covariances according to the normalization factor: each entry should just be the product of the corresponding
        # factors in the joint_normalization_factor array.
        adjusted_covariances = np.diag(self.joint_normalization_factor)
        helper_ones = np.ones((self.num_hidden_dimensions, self.num_hidden_dimensions))
        self.covariance_normalization = (adjusted_covariances.dot(helper_ones)) * \
            (helper_ones.dot(adjusted_covariances))

    def assemble_joint_state(self,
                             transient: TransientVector = None,
                             asoh: AdvancedStateOfHealth = None,
                             joint_covariance: np.ndarray = None) -> Union[np.ndarray, KalmanHiddenState]:
        """
        Method to assemble joint state.

        Args:
            transient: transient vector to be used; if not provided, uses the one stored
            asoh: A-SOH object; if not provided, uses the one stored
            joint_covariance: raw (un-normalized) covariance of the joint state
        """
        if transient is None:
            transient = self.transient
        if asoh is None:
            asoh = self.asoh
        joint = np.hstack((transient.to_numpy(), asoh.get_parameters()))
        # Remember that this is in raw values that need to be normalized to the estimator!
        joint /= self.joint_normalization_factor
        if joint_covariance is None:
            return joint
        # Remember to also adjust the joint state covariance appropriately!
        return KalmanHiddenState(mean=joint, covariance=joint_covariance/self.covariance_normalization)


class JointUKFEstimator(JointOnlineEstimator):
    """
    Class to establish the joint estimator based on UKF.

    Args:
        initial_transient: specifies the initial transient state
        inial_asoh: specifies the initial A-SOH
        initial_control: specifies the initial controls/inputs
        covariance_joint: specifies the raw (un-normalized) covariance of the joint state; it is the preferred method of
            assembling the initial joint state
        covariance_transient: specifies the raw (un-normalized) covariance of the transient state; it is not used if
            covariance_joint was provided
        covariance_asoh: specifies the raw (un-normalized) covariance of the A-SOH; it is not used if covariance_joint
            was provided
        transient_noise: specifies raw (un-normalized) noise covariance of transient update
        asoh_noise: specifies raw (un-normalized) noise covariance of A-SOH
        sensor_noise: specifies noise covariance of measurements
        tuning_params: keyword parameters used to tune the UKF (alpha_param, kappa_param, beta_param)
    """

    def __init__(self,
                 initial_transient: TransientVector,
                 initial_asoh: AdvancedStateOfHealth,
                 initial_control: InputQuantities,
                 covariance_joint: np.ndarray = None,
                 covariance_transient: np.ndarray = None,
                 covariance_asoh: np.ndarray = None,
                 transient_noise: np.ndarray = None,
                 asoh_noise: np.ndarray = None,
                 sensor_noise: np.ndarray = None,
                 normalize_joint_state: bool = False,
                 **tuning_params) -> None:
        # Create interface
        self._init_interface(asoh=initial_asoh,
                             transient=initial_transient,
                             control=initial_control,
                             normalize_joint_state=normalize_joint_state)

        # Create initial joint state
        if covariance_joint is None:
            covariance_joint = block_diag(covariance_transient, covariance_asoh)
        initial_joint = self.interface.assemble_joint_state(joint_covariance=covariance_joint)

        # Creating initial control
        u = ControlVariables(mean=initial_control.to_numpy())

        # Create estimator

        # Remeber to adjust the process noise based on the joint state normalization
        covariance_joint_process_noise = block_diag(transient_noise, asoh_noise)
        covariance_joint_process_noise /= self.interface.covariance_normalization
        self.estimator = UKF(model=self.interface,
                             initial_state=initial_joint,
                             initial_control=u,
                             covariance_process_noise=covariance_joint_process_noise,
                             covariance_sensor_noise=sensor_noise,
                             **tuning_params)

    def _init_interface(self,
                        asoh: AdvancedStateOfHealth,
                        transient: TransientVector,
                        control: InputQuantities,
                        normalize_joint_state: bool = False,) -> None:
        """
        Helper class to initialize the model filter interface
        """
        self.interface = ModelJointUKFInterface(asoh=asoh,
                                                transient=transient,
                                                control=control,
                                                normalize_joint_state=normalize_joint_state)

    @abstractmethod
    def step(self,
             u: InputQuantities,
             y: OutputQuantities) -> Tuple[KalmanOutputMeasurement, KalmanHiddenState]:
        """
        Function to step the estimator, provided new control variables and output measurements.

        Args:
            u: control variables
            y: output measurements

        Returns:
            Corrected estimate of the hidden state of the system
        """
        # Perform the estimation
        estimator_prediction, estimator_hidden = super().step(u=u, y=y)

        # Remember to adequately correct the mean and covariance of the hidden state based on normalization
        estimator_hidden.mean *= self.interface.joint_normalization_factor
        estimator_hidden.covariance *= self.interface.covariance_normalization

        # Update the transient state and A-SOH in the interface
        self.interface.update_from_joint(joint_state=estimator_hidden.mean)
        return estimator_prediction.model_copy(deep=True), estimator_hidden.model_copy(deep=True)
