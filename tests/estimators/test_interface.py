"""Test the interface between model and online estimator"""
import numpy as np

from moirae.estimators.online import ModelFilterInterface, ControlVariables
from moirae.models.ecm import ECMInput


def test_rint_model(rint_parameters):
    asoh, transients, model = rint_parameters
    inputs = ECMInput(time=0., current=1.)
    mint = ModelFilterInterface(
        model=model,
        initial_inputs=inputs,
        initial_asoh=asoh,
        initial_transients=transients,
    )
    assert mint.num_transients == 2  # Hysteresis, SOC
    assert mint.num_hidden_dimensions == 2
    assert mint.num_output_dimensions == 1

    # Ensure the predicted measurement works
    hidden_states = np.array([0., 0.])[None, :]  # a batch of 1 hidden state
    controls = ControlVariables(mean=np.array([0., 1.]))  # time of zero, current of 1 A
    outputs = mint.predict_measurement(
        hidden_states,
        controls,
    )
    assert outputs.shape == (1, 1)  # One batch, one value
    assert np.allclose(outputs, asoh.ocv.get_value(soc=0.) + asoh.r0.get_value(0.) * 1)

    # Ensure the timestep update works too
    new_controls = controls.model_copy(deep=True)
    new_controls.mean[0] = 1.  # Update the time of the second measurement to 1s
    new_hidden_states = mint.update_hidden_states(
        hidden_states, new_controls=new_controls, previous_controls=controls,
    )
    assert new_hidden_states.shape == hidden_states.shape
    assert np.allclose(new_hidden_states, [1. / asoh.q_t.value, 0.])


def test_rint_with_updatable(rint_parameters):
    """Test the model interface when an ASOH parameter is allowed to be udpatable"""
    asoh, transients, model = rint_parameters
    inputs = ECMInput(time=0., current=1.)
    asoh.mark_updatable('r0.base_values')
    assert asoh.num_updatable == 1

    mint = ModelFilterInterface(
        model=model,
        initial_inputs=inputs,
        initial_asoh=asoh,
        initial_transients=transients,
    )
    assert mint.num_transients == 2  # Hysteresis, SOC
    assert mint.num_hidden_dimensions == 3  # Includes r_int
    assert mint.num_output_dimensions == 1

    # Ensure the predicted measurement works
    hidden_states = np.array([0., 0., asoh.r0.get_value(soc=0)])[None, :]  # a batch of 1 hidden state
    controls = ControlVariables(mean=np.array([0., 1.]))  # time of zero, current of 1 A
    outputs = mint.predict_measurement(
        hidden_states,
        controls,
    )
    assert outputs.shape == (1, 1)  # One batch, one value
    assert np.allclose(outputs, asoh.ocv.get_value(soc=0.) + asoh.r0.get_value(0.) * 1)

    # Ensure the timestep update works too
    new_controls = controls.model_copy(deep=True)
    new_controls.mean[0] = 1.  # Update the time of the second measurement to 1s
    new_hidden_states = mint.update_hidden_states(
        hidden_states, new_controls=new_controls, previous_controls=controls,
    )
    assert new_hidden_states.shape == hidden_states.shape
    assert np.allclose(new_hidden_states, [1. / asoh.q_t.value, 0., asoh.r0.get_value(0.)])
