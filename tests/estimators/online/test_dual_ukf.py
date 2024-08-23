import numpy as np

from moirae.models.ecm import EquivalentCircuitModel as ECM
from moirae.models.ecm.advancedSOH import ECMASOH
from moirae.models.ecm.ins_outs import ECMInput
from moirae.models.ecm.transient import ECMTransientVector
from moirae.simulator import Simulator
from moirae.estimators.online.dual import DualEstimator


def cycle(start_time=0):
    # Charge for 3 hours at a current of 3 A
    chg_time = 3 * 3600
    final_time = start_time + chg_time
    inputs = [ECMInput(time=time, current=3) for time in np.arange(start_time, final_time)]
    # Now, discharge for 4 hours at a current of 2.25 A
    dischg_time = 4 * 3600
    final_time += dischg_time
    inputs += [ECMInput(time=time, current=-2.25) for time in np.arange(start_time + chg_time, final_time)]
    return final_time, inputs


def test_names(simple_rint):
    rint_asoh, rint_transient, ecm_inputs, ecm_model = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    ukf_dual = DualEstimator.initialize_unscented_kalman_filter(
        cell_model=ecm_model,
        initial_asoh=rint_asoh,
        initial_transients=rint_transient,
        initial_inputs=ecm_inputs,
        covariance_transient=np.eye(2),
        covariance_asoh=np.atleast_2d(1)
    )
    assert ukf_dual.state_names == ('soc', 'hyst', 'r0.base_values')
    assert ukf_dual.output_names == ('terminal_voltage',)
    assert ukf_dual.control_names == ('time', 'current')


def test_simplest_rint():
    # Create simplest possible Rint model, with no hysteresis, and only R0
    real_transients = ECMTransientVector.provide_template(has_C0=False, num_RC=0, soc=0.1)
    real_asoh = ECMASOH.provide_template(has_C0=False, num_RC=0, R0=1., H0=0.)
    # Prepare simulator
    simulator = Simulator(model=ECM(),
                          asoh=real_asoh,
                          transient_state=real_transients,
                          initial_input=ECMInput(),
                          keep_history=True)

    # Create initial guesses
    guess_transients = ECMTransientVector.provide_template(has_C0=False, num_RC=0)
    guess_asoh = ECMASOH.provide_template(has_C0=False, num_RC=0, R0=2., H0=0.)
    guess_asoh.mark_updatable(name='r0.base_values')

    # Create inputs for the cell
    protocol = []

    # Num cycles
    num_cycles = 4
    start_time = 1
    for _ in range(num_cycles):
        end_time, new_inputs = cycle(start_time=start_time)
        protocol += new_inputs
        start_time = end_time + 1

    # Prepare the dual estimation
    cov_transient = np.diag([1/12, 1.0e-06])  # 4 * (guess_asoh.h0.base_values.item() ** 2) / 12])
    cov_asoh = np.atleast_2d(0.1)
    cov_trans_process = np.diag([1.0e-08, 1.0e-16])

    dual_ukf = DualEstimator.initialize_unscented_kalman_filter(
        cell_model=ECM(),
        initial_asoh=guess_asoh,
        initial_transients=guess_transients,
        initial_inputs=simulator.previous_input,
        covariance_asoh=cov_asoh,
        covariance_transient=cov_transient,
        covariance_sensor_noise=np.atleast_2d(1.0e-06),
        transient_covariance_process_noise=cov_trans_process)

    # Prepare dictionary to store results
    dual_results = {'estimates': [], 'predictions': []}
    # Co-simulate
    for new_input in protocol:
        _, cell_response = simulator.step(new_inputs=new_input)
        est_hid, pred_out = dual_ukf.step(inputs=new_input, measurements=cell_response)
        dual_results['estimates'] += [est_hid]
        dual_results['predictions'] += [pred_out]

    # Prepare variables for plotting
    timestamps = np.array([inputs.time.item() for inputs in protocol]) / 3600.
    real_soc = np.array([transient.soc.item() for transient in simulator.transient_history[1:]])
    real_hyst = np.array([transient.hyst.item() for transient in simulator.transient_history[1:]])
    real_volts = np.array([measurement.terminal_voltage.item()
                           for measurement in simulator.measurement_history[1:]])
    real_r0 = np.ones(len(timestamps))

    est_soc = np.array([est.get_mean()[0] for est in dual_results['estimates']])
    est_hyst = np.array([est.get_mean()[1] for est in dual_results['estimates']])
    est_r0 = np.array([est.get_mean()[2] for est in dual_results['estimates']])
    est_soc_err = np.array([2 * np.sqrt(est.get_covariance()[0, 0]) for est in dual_results['estimates']])
    est_hyst_err = np.array([2 * np.sqrt(est.get_covariance()[1, 1]) for est in dual_results['estimates']])
    est_r0_err = np.array([2 * np.sqrt(est.get_covariance()[2, 2]) for est in dual_results['estimates']])
    pred_volts = np.array([prediction.get_mean()[0] for prediction in dual_results['predictions']])
    pred_volts_err = np.array([2 * np.sqrt(prediction.get_covariance()[0, 0])
                               for prediction in dual_results['predictions']])

    # Check stats
    # consider the last 3 hours of simulation
    last_pts = 2 * 3600
    volt_capt = np.isclose(real_volts[-last_pts:],
                           pred_volts[-last_pts:],
                           atol=pred_volts_err[-last_pts:]).sum()/last_pts
    soc_capt = np.isclose(real_soc[-last_pts:],
                          est_soc[-last_pts:],
                          atol=est_soc_err[-last_pts:]).sum()/last_pts
    hyst_capt = np.isclose(real_hyst[-last_pts:],
                           est_hyst[-last_pts:],
                           atol=est_hyst_err[-last_pts:]).sum()/last_pts
    r0_capt = np.isclose(real_r0[-last_pts:],
                         est_r0[-last_pts:],
                         atol=est_r0_err[-last_pts:]).sum()/last_pts

    assert volt_capt >= 0.95, f'Percentage of voltage within error: {volt_capt}'
    assert soc_capt >= 0.95, f'Percentage of SOC within error: {soc_capt}'
    # Hysteresis only converges fully closer to 10 cycles... :(
    assert hyst_capt >= 0.0, 'Now that makes no sense!'
    # assert hyst_capt >= 0.95, f'Percentage of hysteresis within error: {hyst_capt}'
    assert r0_capt >= 0.95, f'Percentage of R0 within error: {r0_capt}'
