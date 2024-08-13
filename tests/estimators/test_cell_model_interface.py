"""Tests for reducing the cell model to an interface which only uses NumPy functions"""
from pytest import raises
import numpy as np

from moirae.estimators.utils.model import JointCellModelInterface


def test_update_hidden_only(simple_rint):
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint

    cell_function = JointCellModelInterface(
        model=ecm,
        asoh=rint_asoh,
        transients=rint_transient,
        input_template=rint_inputs,
        asoh_inputs=tuple(),
    )

    rint_inputs.current = -np.atleast_2d(1.0)  # Negative current is charging
    new_inputs = rint_inputs.model_copy(deep=True)
    rint_inputs.time = np.atleast_2d([1.0])

    # Test the update function
    hidden_state = cell_function.create_hidden_state(rint_asoh, rint_transient)
    assert np.allclose(hidden_state, [[0.0, 0.0]])
    new_hidden = cell_function.update_hidden_state(
        hidden_states=hidden_state,
        new_control=new_inputs.to_numpy(),
        previous_control=rint_inputs.to_numpy()
    )
    assert np.allclose(new_hidden[0, 0], 1. / 3600 / 10)

    # Test the output function
    output = cell_function.predict_outputs(
        hidden_states=hidden_state,
        new_control=new_inputs.to_numpy()
    )
    expected_voltage = rint_asoh.ocv(soc=0.) + new_inputs.current * rint_asoh.r0.get_value(soc=0.)
    assert np.allclose(output, expected_voltage)


def test_update_batched_inputs(simple_rint):
    """Make sure we get an error if provided batched inputs to the initializer"""
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint
    rint_asoh.mark_updatable('q_t.base_values')

    rint_transient.from_numpy(np.array([[0.0, 0.0], [0.1, 0.0]]))

    with raises(ValueError, match='transient state must be 1. Found: 2'):
        JointCellModelInterface(
            model=ecm,
            asoh=rint_asoh,
            transients=rint_transient,
            input_template=rint_inputs,
            asoh_inputs=('q_t.base_values',),
        )

    rint_asoh.update_parameters(np.array([[10.], [9.], [11.]]), ['q_t.base_values'])
    with raises(ValueError, match='ASOH must be 1. Found: 3'):
        JointCellModelInterface(
            model=ecm,
            asoh=rint_asoh,
            transients=rint_transient,
            input_template=rint_inputs,
            asoh_inputs=('q_t.base_values',),
        )
