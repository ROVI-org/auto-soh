""" Defines basic functionality for Joint UKF Estimator """
from typing import Union

import numpy as np
from scipy.linalg import block_diag

from asoh.models.base import AdvancedStateOfHealth, TransientVector, InputQuantities
from asoh.estimators.online import ControlVariables
from asoh.estimators.online.joint import ModelJointEstimatorInterface, JointOnlineEstimator
from asoh.estimators.online.general.kalman import KalmanHiddenState
from asoh.estimators.online.general.kalman.unscented import UnscentedKalmanFilter as UKF


class ModelJointUKFInterface(ModelJointEstimatorInterface):
    """
    Class to interface model with Joint UKF
    """

    def assemble_joint_state(self,
                             transient: TransientVector = None,
                             asoh: AdvancedStateOfHealth = None,
                             joint_covariance: np.ndarray = None) -> Union[np.ndarray, KalmanHiddenState]:
        """
        Method to assemble joint state
        """
        if transient is None:
            transient = self.transient.to_numpy()
        if asoh is None:
            asoh = self.asoh
        joint = np.hstack((transient.to_numpy(), asoh.get_parameters()))
        if joint_covariance is None:
            return joint
        return KalmanHiddenState(mean=joint, covariance=joint_covariance)


class JointUKFEstimator(JointOnlineEstimator):
    """
    Class to establish the joint estimator based on UKF.

    Args:
        initial_transient: specifies the initial transient state
        inial_asoh: specifies the initial A-SOH
        initial_control: specifies the initial controls/inputs
        covariance_joint: specifies the covariance of the joint state; it is the preferred method of assembling the
            initial joint state
        covariance_transient: specifies the covariance of the transient state; it is not used if covariance_joint was
            provided
        covariance_asoh: specifies the covariance of the A-SOH; it is not used if covariance_joint was provided
        transient_noise: specifies noise covariance of transient update
        asoh_noise: specifies noise covariance of A-SOH
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
                 **tuning_params) -> None:
        # Create interface
        self._init_interface(asoh=initial_asoh, transient=initial_transient, control=initial_control)

        # Create initial joint state
        if covariance_joint is None:
            covariance_joint = block_diag(covariance_transient, covariance_asoh)
        initial_joint = self.interface.assemble_joint_state(covariance_joint=covariance_joint)

        # Creating initial control
        u = ControlVariables(mean=initial_control.to_numpy())

        # Create estimator
        self.estimator = UKF(model=self.interface,
                             initial_state=initial_joint,
                             initial_control=u,
                             covariance_process_noise=block_diag(transient_noise, asoh_noise),
                             covariance_sensor_noise=sensor_noise,
                             **tuning_params)

    def _init_interface(self,
                        asoh: AdvancedStateOfHealth,
                        transient: TransientVector,
                        control: InputQuantities) -> None:
        """
        Helper class to initialize the model filter interface
        """
        self.interface = ModelJointEstimatorInterface(asoh=asoh,
                                                      transient=transient,
                                                      control=control)
