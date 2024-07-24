""" Testing ECM physics and evolution """
import numpy as np
from pytest import fixture

from moirae.models.ecm import EquivalentCircuitModel as ECM
from moirae.models.ecm import (ECMASOH,
                               ECMInput,
                               ECMTransientVector
                               )
from moirae.models.ecm.simulator import ECMSimulator


@fixture
def rint_const() -> ECMSimulator:
    rint_asoh = ECMASOH.provide_template(has_C0=False, num_RC=0)
    # Removing hysteresis
    rint_asoh.h0.base_values = 0
    return ECMSimulator(asoh=rint_asoh, keep_history=True)


@fixture
def hyst_only() -> ECMSimulator:
    hyst_asoh = ECMASOH.provide_template(has_C0=False, num_RC=0)
    # Removing R0
    hyst_asoh.r0.base_values = 0
    # Setting hysteresis to known value and speeding up its rate of asymptotic approach
    hyst_asoh.h0.base_values = 3 * np.pi / 10
    hyst_asoh.h0.gamma = 15
    return ECMSimulator(asoh=hyst_asoh, keep_history=True)


@fixture
def c0_asoh() -> ECMASOH:
    asoh = ECMASOH.provide_template(has_C0=True, num_RC=0)
    # Remove R0
    asoh.r0.base_values = 0
    # Set CE to known value
    asoh.ce = 3 * np.pi / 10
    return asoh


@fixture
def rc_only() -> ECMSimulator:
    asoh = ECMASOH.provide_template(has_C0=False, num_RC=1)
    # Remove R0
    asoh.r0.base_values = 0
    # Removing hysteresis
    asoh.h0.base_values = 0
    # Setting RC values
    asoh.rc_elements[0].r.base_values = 10
    asoh.rc_elements[0].c.base_values = 3 * np.pi
    # I want the simulator to start at an SOC of 0.5
    start_transient = ECMTransientVector.provide_template(has_C0=False, num_RC=1, soc=0.5, i_rc=np.array([0.]))
    return ECMSimulator(asoh=asoh, transient_state=start_transient, keep_history=True)


def test_current_integration_CE_C0(c0_asoh) -> None:
    # Define first two transient states
    ipt1 = ECMInput(time=1, current=2)
    ipt2 = ECMInput(time=3, current=5)
    # Define a starting transient state
    tns1 = ECMTransientVector(soc=0, q0=0)
    # Now, let's do a step assuming constant current
    tns2_const = ECM().update_transient_state(current_input=ipt2,
                                              transient_state=tns1,
                                              asoh=c0_asoh,
                                              previous_input=ipt1)
    vt2_const = ECM().calculate_terminal_voltage(inputs=ipt2, transient_state=tns2_const, asoh=c0_asoh)
    expected_q0 = (ipt2.time - ipt1.time) * ipt1.current
    expected_soc = (3 * np.pi / 10) * expected_q0 / c0_asoh.q_t.value
    expected_vt2 = c0_asoh.ocv(expected_soc) + (expected_q0 / c0_asoh.c0.get_value(soc=expected_soc))
    assert np.allclose(expected_q0, tns2_const.q0), \
        'Expected constant q0 of %1.3f, but calculated %1.3f!' % (expected_q0, tns2_const.q0)
    assert np.allclose(expected_soc, tns2_const.soc), \
        'Expected constant SOC of %1.3f, but calculated %1.3f!' % (expected_soc, tns2_const.soc)
    assert np.allclose(expected_vt2, vt2_const.terminal_voltage), \
        'Expected constant Vt of %1.3f, but calculated %1.3f' % (expected_vt2, vt2_const.terminal_voltage)
    # Now, step assuming linear current
    tns2_lin = ECM(current_behavior='linear').update_transient_state(current_input=ipt2,
                                                                     transient_state=tns1,
                                                                     asoh=c0_asoh,
                                                                     previous_input=ipt1)
    vt2_lin = ECM().calculate_terminal_voltage(inputs=ipt2, transient_state=tns2_lin, asoh=c0_asoh)
    # Trapezoid area for q0
    expected_q0 = (ipt2.time - ipt1.time) * (ipt2.current + ipt1.current) / 2
    expected_soc = (3 * np.pi / 10) * expected_q0 / c0_asoh.q_t.value
    expected_vt2 = c0_asoh.ocv(expected_soc) + (expected_q0 / c0_asoh.c0.get_value(soc=expected_soc))
    assert np.allclose(expected_q0, tns2_lin.q0), \
        'Expected constant q0 of %1.3f, but calculated %1.3f!' % (expected_q0, tns2_lin.q0)
    assert np.allclose(expected_soc, tns2_lin.soc), \
        'Expected constant SOC of %1.3f, but calculated %1.3f!' % (expected_soc, tns2_lin.soc)
    assert np.allclose(expected_vt2, vt2_lin.terminal_voltage), \
        'Expected constant Vt of %1.3f, but calculated %1.3f' % (expected_vt2, vt2_lin.terminal_voltage)


def test_rint_const(rint_const) -> None:
    # Get value of max capacity in Amp-hour
    Qt = rint_const.asoh.q_t.value
    # Use charge rate of 1 C
    current = Qt / 3600
    # Create first input (remember that, after stepping through it, nothing should have happened to the SOC)
    input0 = ECMInput(time=1, current=current)
    # Create next input to match 10% SOC (happens after 6 minutes at 1C rate)
    input1 = ECMInput(time=361, current=current)
    # Evolve
    rint_const.evolve(inputs=[input0, input1])
    assert rint_const.transient_history[1].soc == 0, 'SOC wrongly changed in first step!'
    assert np.allclose(rint_const.transient_history[-1].soc, 0.1), \
        'Wrong SOC at the end of constant Rint! Should be 0.1 but is %1.3f' % rint_const.transient_history[-1].soc
    # Now, let's double check the voltage values are correct. For that, we will need the OCV and the R0 value
    r0 = rint_const.asoh.r0
    ocv = rint_const.asoh.ocv
    vt0 = ocv(0) + (r0.get_value(soc=0) * current)
    vt1 = ocv(0.1) + (r0.get_value(soc=0.1) * current)
    assert rint_const.measurement_history[1].terminal_voltage == vt0, 'Wrong initial voltage!'
    assert np.allclose(rint_const.measurement_history[-1].terminal_voltage, vt1), 'Wrong final voltage!'


def test_hyst_only(hyst_only) -> None:
    # Get value of max capacity in Amp-hour
    Qt = hyst_only.asoh.q_t.value
    # Use a C/10 charge rate to make sure we will reach max hysteresis
    current = 0.1 * (Qt / 3600)
    # Now, prepare charging time
    chg_time = np.arange((10 * 3600) + 1) + 1
    chg_currs = [current] * len(chg_time)
    chg_ins = [ECMInput(time=t, current=i) for t, i in zip(chg_time, chg_currs)]
    # Evolve charge
    hyst_only.evolve(inputs=chg_ins)
    # Check that we reached 100% SOC
    assert np.allclose(hyst_only.transient_history[-1].soc, 1), \
        'End of charge SOC should be 1.0, instead, it is %1.5f!' % (hyst_only.transient_history[-1].soc)
    # Check final voltage is correct
    vt = hyst_only.asoh.ocv(1.0) + (3 * np.pi / 10)
    assert np.allclose(hyst_only.measurement_history[-1].terminal_voltage, vt), \
        'End of charge terminal voltage should be %1.3f, but instead, it is %1.3f!' % \
        (vt, hyst_only.measurement_history[-1].terminal_voltage)
    # Briefly rest the cell
    rest_time = np.arange(120) + (chg_time[-1] + 0.001)
    rest_curr = [0] * len(rest_time)
    rest_ins = [ECMInput(time=t, current=i) for t, i in zip(rest_time, rest_curr)]
    hyst_only.evolve(inputs=rest_ins)
    # Now, let's discharge and make sure we revert the hysteresis
    dischg_time = np.arange((10 * 3600) + 1) + (rest_time[-1] + 0.001)
    dischg_currs = [-current] * len(dischg_time)
    dischg_ins = [ECMInput(time=t, current=i) for t, i in zip(dischg_time, dischg_currs)]
    # Evolve discharge
    hyst_only.evolve(inputs=dischg_ins)
    assert np.allclose(hyst_only.transient_history[-1].soc, 0, atol=1e-7), \
        'End of discharge SOC should be 0.0, but instead, it is %1.3f!' % (hyst_only.transient_history[-1].soc)
    vt = hyst_only.asoh.ocv(0.0) - (3 * np.pi / 10)
    assert np.allclose(hyst_only.measurement_history[-1].terminal_voltage, vt), \
        'End of discharge terminal voltage should be %1.3f, but instead, it is %1.3f!' % \
        (vt, hyst_only.measurement_history[-1].terminal_voltage)


def test_RC(rc_only) -> None:
    # Recall that the simulator starts at an SOC of 0.5 with no RC overpotential, which should correspond to a terminal
    # voltage of 3.5 V
    v0 = rc_only.measurement_history[0].terminal_voltage
    assert np.allclose(v0, 3.5), 'Starting teminal voltage should be 3.5 V, but instead, it is %1.3f V!' % v0
    # Let's retrieve valuable info to decide charging and discharging protocols
    Qt = rc_only.asoh.q_t.amp_hour
    # We will simulate a 30 minute charge at a C/5 rate
    curr = Qt / 5
    chg_time = np.arange(1800) + 1
    chg_curr = [curr] * len(chg_time)
    chg_ins = [ECMInput(time=t, current=i) for t, i in zip(chg_time, chg_curr)]
    rc_only.evolve(inputs=chg_ins)
    # Let us retrieve the SOCs and the corresponding OCVs to check the voltage convergence
    chg_socs = [transient.soc for transient in rc_only.transient_history[1:]]
    chg_ocvs = rc_only.asoh.ocv(chg_socs)
    vrc_calc = np.array([out.terminal_voltage for out in rc_only.measurement_history[1:]]) - chg_ocvs
    # The RC element should asymptotically approach a stage where the current through the resistive component is equal
    # to the total current flowing through the system. Let's them calculate the expected voltage drop across the RC
    r_rc = 10
    c_rc = 3 * np.pi
    tau_rc = r_rc * c_rc
    expected_vrc = (curr * r_rc) * (1.0 - np.exp(-chg_time / tau_rc))
    vrc_err = vrc_calc - expected_vrc
    vrc_rmse = np.sqrt(np.mean((vrc_err ** 2)))
    assert np.allclose(vrc_calc, expected_vrc), 'RMSE of V_RC calculations: %1.5f V!' % vrc_rmse
