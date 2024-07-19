"""
Here, we have a collection of joint estimators for battery cell models.

In joint estimation, the physical transident state and the model health variables are concatenated in a single vector,
which is updated by a single online estimator at every new time step. However, the interfaces between model and
estimator must be able to:
    1. break up a joint state into transient and A-SOH components
    2. convert numpy representations of transient vectors, A-SOH components, and inputs/controls to the appropriate
        objects
    3. convert transient vectors, A-SOH, and input/control objects to numpy arrays and to the adequate joint states
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
        transient: initial transiden hidden state of the syste
        control: initial control to the system
        normalize_asoh: determines if the numerical representation of the A-SOH  should be made up of raw values, or
            normalized, so that the filter only deals with values close to 1.
    """
    def __init__(self,
                 asoh: AdvancedStateOfHealth,
                 transient: TransientVector,
                 control: InputQuantities,
                 normalize_asoh: bool = False) -> None:
        self.asoh = asoh.model_copy(deep=True)
        self.transient = transient.model_copy(deep=True)
        self.control = control.model_copy(deep=True)
        self.is_joint_normalized = normalize_asoh
        joint_normalization_factor = np.ones(self.num_hidden_dimensions)
        if normalize_asoh:
            joint_normalization_factor[len(transient):] = asoh.get_parameters()
            # Special attention needs to be paid to cases whewre the initial provided value is 0.0. In these cases,
            # the normalization factor remains equal to 1. (variable is "un-normalized" and treated as raw.)
            joint_normalization_factor = np.where(joint_normalization_factor == 0, 1., joint_normalization_factor)
        self.joint_normalization_factor = joint_normalization_factor

    @property
    @abstractmethod
    def num_hidden_dimensions(self) -> int:
        """
        Outputs expected dimensionality of hidden state, which should include the transient state and the A-SOH
        """
        return self.asoh.num_updatable + len(self.transient)

    @property
    @abstractmethod
    def num_output_dimensions(self) -> int:
        """ Outputs expected dimensionality of output measurements """
        pass

    @abstractmethod
    def assemble_joint_state(self,
                             transient: TransientVector = None,
                             asoh: AdvancedStateOfHealth = None) -> HiddenState:
        """
        Method to assemble joint state
        """
        if transient is None:
            transient = self.transient.to_numpy()
        if asoh is None:
            asoh = self.asoh
        joint = np.hstack((transient.to_numpy(), asoh.get_parameters()))
        # Remember that this is in raw values that need to be normalized to the estimator!
        joint /= self.joint_normalization_factor
        return HiddenState(mean=joint)

    def update_from_joint(self, joint_state: np.ndarray) -> None:
        """
        Method that updates the transient state and the A-SOH from a numpy representation of the joint state
        """
        if len(joint_state) != self.num_hidden_dimensions:
            raise ValueError('Joint state has %d dimensions, but it should have %d!' %
                             (len(joint_state), self.num_hidden_dimensions))
        # Remember the joint state is in its "normalized" form, and needs to be converted to raw values!
        raw_values_joint = joint_state * self.joint_normalization_factor
        # Make sure that the A-SOH components are always positive
        raw_values_joint[-self.asoh.num_updatable:] = np.where(raw_values_joint[-self.asoh.num_updatable:] < 0, 1.0e-16,
                                                               raw_values_joint[-self.asoh.num_updatable:])
        self.transient.from_numpy(raw_values_joint[:len(self.transient)])
        self.asoh.update_parameters(raw_values_joint[-self.asoh.num_updatable:])

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
        normalize_asoh: determines if the numerical representation of the A-SOH  should be made up of raw values, or
            normalized, so that the filter only deals with values close to 1.
    """

    def __init__(self,
                 initial_transient: TransientVector,
                 initial_asoh: AdvancedStateOfHealth,
                 initial_control: InputQuantities,
                 normalize_asoh: bool = False):
        self.interface = ModelJointEstimatorInterface(asoh=initial_asoh,
                                                      transient=initial_transient,
                                                      control=initial_control,
                                                      normalize_asoh=normalize_asoh)
        self.estimator = OnlineEstimator(initial_state=self.interface.assemble_joint_state(),
                                         initial_control=self.u)

    @abstractmethod
    def step(self,
             u: InputQuantities,
             y: OutputQuantities) -> Tuple[OutputMeasurements, HiddenState]:
        """
        Function to step the estimator, provided new control variables and output measurements.

        Args:
            u: control variables
            y: output measurements

        Returns:
            Corrected estimate of the hidden state of the system
        """
        # Convert the model related quantities to the filter related quantities
        u_estimator = ControlVariables(mean=u.to_numpy())
        y_estimator = OutputMeasurements(mean=y.to_numpy())
        # Get the measurement predictions and hidden state from estimator
        estimator_prediction, estimator_hidden = self.estimator.step(u=u_estimator, y=y_estimator)

        # Update the transient state and A-SOH in the interface
        self.interface.update_from_joint(joint_state=estimator_hidden.mean)
        return estimator_prediction.model_copy(deep=True), estimator_hidden.model_copy(deep=True)
