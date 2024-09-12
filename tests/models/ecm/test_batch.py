""" Testing batching capabilities """
from typing import Tuple

import numpy as np
from pytest import fixture

from moirae.models.ecm import (ECMASOH,
                               ECMInput,
                               ECMTransientVector)
from moirae.models.ecm import EquivalentCircuitModel as ECM
from moirae.simulator import Simulator


@fixture
def rint() -> Tuple[ECMTransientVector, ECMASOH, Simulator]:
    soc = np.array([0, 0.25, 0.75])
    hyst = np.array([-1, 0, 1])
    transient = ECMTransientVector(soc=soc, hyst=hyst)
    asoh = ECMASOH.provide_template(has_C0=False, num_RC=0)
    # Make hysteresis approach asymptotic value fast
    asoh.h0.gamma = 100
    simulator = Simulator(cell_model=ECM(),
                          asoh=asoh,
                          transient_state=transient,
                          initial_input=ECMInput(),
                          keep_history=True)
    return transient, asoh, simulator


@fixture
def pngv() -> Tuple[ECMTransientVector, ECMASOH, Simulator]:
    soc = np.array([0, 0.25, 0.75])
    hyst = np.array([-1, 0, 1])
    q0 = np.array([0., 1., 2])
    i_rc = np.arange(6, dtype=float).reshape((3, 2))
    transient = ECMTransientVector(soc=soc, hyst=hyst, q0=q0, i_rc=i_rc)
    asoh = ECMASOH.provide_template(has_C0=True, num_RC=2)
    simulator = Simulator(cell_model=ECM(),
                          asoh=asoh,
                          transient_state=transient,
                          initial_input=ECMInput(),
                          keep_history=True)
    return transient, asoh, simulator


def test_rint(rint) -> None:
    _, asoh, simulator = rint
    # Get total capacity
    Qt = asoh.q_t.amp_hour
    # Provide enough current for only 0.25 SOC charge over 1 hour (C/4 rate)
    current = Qt / 4.
    # Simulator was initialized with a current value of 0, so let's change that
    simulator.previous_input.current = current
    # Prepare next input, one hour later...
    new_input = ECMInput(time=3600, current=current)
    simulator.evolve([new_input])
    # Validate
    assert np.allclose([0.25, 0.5, 1.], simulator.transient.soc[:, 0]), \
        f'Wrong Rint SOCs: {simulator.transient.soc}'
    assert simulator.transient.hyst.shape == (3, 1), \
        f'Hysteresis batching did not work: {simulator.transient.hyst.shape}'
    # hysteresis should be at the max, as it approaches the true value very quickly
    assert np.allclose(asoh.h0.base_values, simulator.transient.hyst), \
        f'Wrong hysteresis values: {simulator.transient.hyst}'


def test_pngv(pngv) -> None:
    transient, asoh, simulator = pngv
    # Get initial q0 values
    initial_q0 = transient.q0.copy()
    # Get total capacity
    Qt = asoh.q_t.amp_hour
    # Provide enough current for only 0.25 SOC charge over 1 hour (C/4 rate)
    current = Qt / 4.
    # Simulator was initialized with a current value of 0, so let's change that
    simulator.previous_input.current = current
    # Prepare next input, one hour later...
    new_input = ECMInput(time=3600, current=current)
    simulator.evolve([new_input])
    # Validate
    assert np.allclose([0.25, 0.5, 1.], simulator.transient.soc.copy().flatten()), \
        f'Wrong Rint SOCs: {simulator.transient_history.soc}'
    assert np.allclose(initial_q0 + (3600 * current), simulator.transient.q0), \
        f'Wrong q0 evolution: {simulator.transient.q0}'
    assert simulator.transient.i_rc.copy().shape == (3, 2), \
        f'Wrong i_rc shape: {simulator.transient.i_rc.shape}'
    # at this point, the currents should all be pretty close to the total current
    assert np.allclose(current, simulator.transient.i_rc), \
        f'Wrong i_rc values: {simulator.transient.i_rc}'


def test_numpy_operations(pngv):
    transient, _, _ = pngv
    # Make sure the to_numpy method will output the right shape and values
    soc = np.array([0, 0.25, 0.75])
    hyst = np.array([-1, 0, 1])
    q0 = np.array([0., 1., 2])
    i_rc = np.arange(6, dtype=float).reshape((3, 2))
    real_vals = np.hstack((np.atleast_2d(soc).T, np.atleast_2d(q0).T, i_rc, np.atleast_2d(hyst).T))
    assert real_vals.shape == transient.to_numpy().shape, f'Wrong shape: {transient.to_numpy().shape}'
    assert np.allclose(real_vals, transient.to_numpy()), f'Wrong original values: {transient.to_numpy()}'
    # Now, update from a numpy array
    new_values = np.arange(15).reshape((5, 3)).T
    transient.from_numpy(new_values)
    assert np.allclose(np.arange(3), transient.soc.copy().flatten()), f'Wrong SOC update: {transient.soc.copy()}'
    assert np.allclose(3 + np.arange(3), transient.q0.copy().flatten()), f'Wrong q0 update: {transient.q0.copy()}'
    assert np.allclose(6 + np.arange(6).reshape((2, 3)).T, transient.i_rc.copy()), \
        f'Wrong i_rc update: {transient.i_rc.copy()}'
    assert np.allclose(12 + np.arange(3), transient.hyst.copy().flatten()), \
        f'Wrong hyst update: {transient.hyst.copy()}'
    assert np.allclose(new_values, transient.to_numpy()), f'Wrong full update: {transient.to_numpy()}'


def test_broadcasting(rint):

    # Create a model
    ecm = ECM()

    # Create inputs
    input0 = ECMInput(time=0, current=0)
    input1 = ECMInput(time=0.1, current=2)

    # Get transient and A-SOH
    transient = ECMTransientVector.provide_template(has_C0=False, num_RC=0)
    asoh = ECMASOH.provide_template(has_C0=False, num_RC=0)

    # Make parts of the asoh updatable
    asoh.mark_updatable(name='r0.base_values')
    asoh.mark_updatable(name='h0.base_values')

    # Create batched copies
    trans_batch = transient.model_copy(deep=True)
    asoh_batch = asoh.model_copy(deep=True)

    # Batch of 5
    batch_size = 5
    trans_vals = np.zeros((batch_size, 2))
    trans_vals[:, 0] = np.linspace(0, 1, batch_size)
    trans_batch.from_numpy(trans_vals)
    # Update each A-SOH element separately, as we want R0 to have a full interpolation
    asoh_batch.h0.base_values = 0.01 * np.arange(batch_size).reshape((-1, 1))
    asoh_batch.r0.base_values = 0.01 * np.arange(2 * batch_size).reshape((-1, 2))

    # Now, mix and match the updates and make sure they go through
    # Trans: unbatched; ASOH: unbatched
    new_trans = ecm.update_transient_state(previous_inputs=input0,
                                           new_inputs=input1,
                                           transient_state=transient,
                                           asoh=asoh)
    volt = ecm.calculate_terminal_voltage(new_inputs=input1, transient_state=transient, asoh=asoh)
    assert new_trans.batch_size == 1
    assert new_trans.to_numpy().shape[1] == 2
    assert volt.batch_size == 1
    assert volt.to_numpy().shape[1] == 1
    # Trans: batched; ASOH: unbatched
    new_trans = ecm.update_transient_state(previous_inputs=input0,
                                           new_inputs=input1,
                                           transient_state=trans_batch,
                                           asoh=asoh)
    volt = ecm.calculate_terminal_voltage(new_inputs=input1, transient_state=trans_batch, asoh=asoh)
    assert new_trans.batch_size == batch_size
    assert new_trans.to_numpy().shape[1] == 2
    assert volt.batch_size == batch_size
    assert volt.to_numpy().shape[1] == 1
    # Trans: unbatched; ASOH: batched
    new_trans = ecm.update_transient_state(previous_inputs=input0,
                                           new_inputs=input1,
                                           transient_state=transient,
                                           asoh=asoh_batch)
    volt = ecm.calculate_terminal_voltage(new_inputs=input1, transient_state=transient, asoh=asoh_batch)
    assert new_trans.batch_size == batch_size
    assert new_trans.to_numpy().shape[1] == 2
    assert volt.batch_size == batch_size
    assert volt.to_numpy().shape[1] == 1
    # Trans: batched; ASOH: batched
    new_trans = ecm.update_transient_state(previous_inputs=input0,
                                           new_inputs=input1,
                                           transient_state=trans_batch,
                                           asoh=asoh_batch)
    volt = ecm.calculate_terminal_voltage(new_inputs=input1, transient_state=trans_batch, asoh=asoh_batch)
    assert new_trans.batch_size == batch_size
    assert new_trans.to_numpy().shape[1] == 2
    assert volt.batch_size == batch_size
    assert volt.to_numpy().shape[1] == 1
