"""
Definition of the Joint UKF Estimator for an ECM model
"""
from functools import cached_property
from typing import Literal

import numpy as np

from asoh.models.ecm import EquivalentCircuitModel as ECM
from asoh.models.ecm.advancedSOH import ECMASOH
from asoh.models.ecm.transient import ECMTransientVector
from asoh.models.ecm.ins_outs import ECMInput
from asoh.estimators.online import ControlVariables
from asoh.estimators.online.joint.unscented import ModelJointUKFInterface, JointUKFEstimator


# TODO (vventuri): how do we denoise SOC, Qt, R0?
class ECMJointUKFInterface(ModelJointUKFInterface):
    """
    Class to help interface the ECM model with the Joint UKF Estimator

    Args:
        asoh: initial Advanced State of Health (A-SOH) of the system
        transient: initial transiden hidden state of the syste
        control: initial control to the system
        current_behavior: determines the current behavior between time steps
    """

    def __init__(self,
                 asoh: ECMASOH,
                 transient: ECMTransientVector,
                 control: ECMInput,
                 current_behavior: Literal['constant', 'linear'] = 'constant') -> None:
        self.asoh = asoh.model_copy(deep=True)
        self.transient = transient.model_copy(deep=True)
        self.control = control.model_copy(deep=True)
        self.output = ECM.calculate_terminal_voltage(new_input=control, transient_state=transient, asoh=asoh)
        self.current_behavior = current_behavior

    @cached_property
    def num_output_dimensions(self) -> int:
        """ Outputs expected dimensionality of output measurements """
        return len(self.output)

    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: ControlVariables,
                             new_controls: ControlVariables) -> np.ndarray:
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
        # Start by assembling the new hidden states we will return
        new_hidden = []

        # Convert the controls to ECMInputs
        self.control.from_numpy(previous_controls.mean)
        previous_input = self.control.model_copy(deep=True)
        self.control.from_numpy(new_controls.mean)
        new_input = self.control.model_copy(deep=True)

        # Update each transient state, joint state by joint state
        for joint_state in hidden_states:
            self.update_from_joint(joint_state=joint_state)
            new_transient = ECM.update_transient_state(new_input=new_input,
                                                       transient_state=self.transient,
                                                       asoh=self.asoh,
                                                       previous_input=previous_input,
                                                       current_behavior=self.current_behavior)
            new_hidden += [new_transient.to_numpy()]

        return np.array(new_hidden)

    def predict_measurement(self,
                            hidden_states: np.ndarray,
                            controls: ControlVariables) -> np.ndarray:
        """
        Function to predict measurement from the hidden state

        Args:
            hidden_states: current joint states of the system as a numpy.ndarray object
            controls: controls to be used for predicting outputs

        Returns:
            pred_measurement: predicted measurements as a numpy.ndarray object
        """
        # Start by assembling the predictions
        pred_measurement = []

        # Convert the controls
        self.control.from_numpy(controls.mean)
        ecm_input = self.control.model_copy(deep=True)

        # For each joint state, update transient and A-SOH and calculate output
        for joint_state in hidden_states:
            self.update_from_joint(joint_state=joint_state)
            ecm_out = ECM.calculate_terminal_voltage(new_input=ecm_input,
                                                     transient_state=self.transient,
                                                     asoh=self.asoh)
            pred_measurement += [ecm_out.to_numpy()]

        return np.array(pred_measurement)


# TODO (vventuri): how do we denoise SOC, Qt, R0?
class ECMJointUKF(JointUKFEstimator):
    """
    Class to define the ECM Joint UKF Estimator. Since all the main updates are performed in the interface, and the
    parent JointUKFEstimator already takes care of initialization and stepping, there is nothing to do here.
    """
    pass

    def _init_interface(self,
                        asoh: ECMASOH,
                        transient: ECMTransientVector,
                        control: ECMInput) -> None:
        """
        Helper class to initialize the model filter interface
        """
        self.interface = ECMJointUKFInterface(asoh=asoh,
                                              transient=transient,
                                              control=control)