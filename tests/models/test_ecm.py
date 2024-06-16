import numpy as np

from asoh.models.ecm import SingleResistorModel, ECMState, ECMControl


def test_rint():
    # Create the initial states of a battery modeled by a 1 Ohm resistor in series
    #  And an OCV = 1 + 0.5 * SOC
    model = SingleResistorModel()
    state = ECMState(r_serial=1., ocv_params=(1., 0.5))

    # Test the model state functions
    assert np.isclose(state.state, [0.]).all()  # Charge is zero
    assert np.isclose(state.soh, []).all()  # There are no health full_params

    state.health_params = ('ocv_params',)
    assert np.isclose(state.full_state, [0., 1., 0.5]).all()  # Includes both charge, then OCV full_params

    # Make sure it steps appropriately
    dx = model.dx(state, ECMControl(current=0.))
    assert np.isclose(dx, [0.]).all()

    dx = model.dx(state, ECMControl(current=1.))
    assert np.isclose(dx, [1. / 3600.]).all()

    # Test the output function
    output = model.output(state, ECMControl(current=0.))
    assert output.terminal_voltage == 1.

    state.charge = 0.5
    output = model.output(state, ECMControl(current=1.))
    assert output.terminal_voltage == 2.25  # 1 V from constant OCV, 0.25 from 0.5 SOC, 1 from the 1 A current and 1 Ohm r_serial


def test_state_model_update():
    state = ECMState(r_serial=1, ocv_params=(1., 0.5), health_params=('r_serial', 'ocv_params'))
    assert np.isclose(state.full_state, [0., 1., 1., 0.5]).all()

    state.set_full_state([0.1, 1.1, 0.9, 0.6])
    assert np.isclose(state.full_state, [0.1, 1.1, 0.9, 0.6]).all()

    state.set_state([0.2])
    assert np.isclose(state.full_state, [0.2, 1.1, 0.9, 0.6]).all()

    state.set_soh([1.2, 0.8, 0.7])
    assert np.isclose(state.full_state, [0.2, 1.2, 0.8, 0.7]).all()


def test_update():
    model = SingleResistorModel()
    state = ECMState(r_serial=1., ocv_params=(1., 0.5))
    model.update(state, ECMControl(current=1.), 4.)
    assert state.charge == 4. / 3600
