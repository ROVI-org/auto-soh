"""
Here, we have a collection of joint estimators for battery cell models.

In joint estimation, the physical transident state and the model health variables are concatenated in a single vector,
which is updated by a single online estimator at every new time step.
"""
from abc import abstractmethod
from typing import Union, Tuple, List

import numpy as np

from asoh.models.base import AdvancedStateOfHealth, TransientVector, InputQuantities, OutputQuantities
from asoh.estimators.online import (OnlineEstimator,
                                    ModelFilterInterface,
                                    ControlVariables,
                                    HiddenState,
                                    OutputMeasurements)


class ModelJointEstimatorInterface(ModelFilterInterface):
    """
    Interface for a joint estimator.

    Args:
        asoh: initial Advanced State of Health (A-SOH) of the system
    """
    def __init__(self,
                 asoh: AdvancedStateOfHealth) -> None:
        self.asoh = asoh.model_copy(deep=True)

    @property
    @abstractmethod
    def num_hidden_dimensions(self) -> int:
        """
        Outputs expected dimensionality of hidden state, which should include the transient state and the A-SOH
        """
        return self.asoh.num_updatable

    @property
    @abstractmethod
    def num_output_dimensions(self) -> int:
        """ Outputs expected dimensionality of output measurements """
        pass

    @abstractmethod
    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: Union[ControlVariables, List[ControlVariables]],
                             new_controls: Union[ControlVariables, List[ControlVariables]]) -> np.ndarray:
        """
        Function that updates the hidden state based on the control variables provided.

        Args:
            hidden_states: current joint states of the system as a numpy.ndarray object
            previous_control: controls at the time the hidden states are being reported. If provided as a list, each
                entry must correspond to one of the hidden states provided. Otherwise, assumes same control for all
                hidden states.
            new_control: new controls to be used in the hidden state update. If provided as a list, each entry must
                correspond to one of the hidden states provided. Otherwise, assumes same control for all hidden states.

        Returns:
            new_hidden: updated hidden states as a numpy.ndarray object
        """
        pass

    @abstractmethod
    def predict_measurement(self,
                            hidden_states: np.ndarray,
                            controls: Union[ControlVariables, List[ControlVariables]]) -> np.ndarray:
        """
        Function to predict measurement from the hidden state

        Args:
            hidden_states: current joint states of the system as a numpy.ndarray object
            controls: controls to be used for predicting outputs

        Returns:
            pred_measurement: predicted measurements as a numpy.ndarray object
        """
        pass


class JointOnlineEstimator(OnlineEstimator):
    """
    Class that defines the joint online estimator.

    Args:
        initial_transient: specifies the initial transient state
        inial_asoh: specifies the initial A-SOH
        initial_control: specifies the initial controls/inputs
    """

    def __init__(self,
                 initial_transient: Union[HiddenState, TransientVector],
                 initial_asoh: AdvancedStateOfHealth,
                 initial_control: Union[ControlVariables, InputQuantities]):
        self.state = initial_transient.model_copy(deep=True)
        self.asoh = initial_asoh.model_copy(deep=True)
        self.u = initial_control.model_copy(deep=True)
        self.interface = ModelJointEstimatorInterface(asoh=initial_asoh)
        # Prepare the joint state
        joint_state = initial_asoh.get_parameters().copy()
        if isinstance(initial_transient, TransientVector):
            joint_state = np.hstack((initial_transient.to_numpy(), joint_state))
        elif isinstance(initial_transient, HiddenState):
            joint_state = np.hstack((initial_transient.get_mean(), joint_state))
        self.joint_state = HiddenState(mean=joint_state)
        # Initialize the actual estimator
        self.estimator = OnlineEstimator(initial_state=self.joint_state, initial_control=self.u)

    @abstractmethod
    def step(self,
             u: Union[ControlVariables, InputQuantities],
             y: Union[OutputMeasurements, OutputQuantities]) -> Tuple[OutputMeasurements, HiddenState]:
        """
        Function to step the estimator, provided new control variables and output measurements.

        Args:
            u: control variables
            y: output measurements

        Returns:
            Corrected estimate of the hidden state of the system
        """
        return self.estimator.step(u=u, y=y)
