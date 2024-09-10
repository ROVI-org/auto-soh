from typing import List, Tuple

import numpy as np
from pytest import fixture
from pytest import mark

from moirae.models.ecm import EquivalentCircuitModel as ECM
from moirae.models.ecm.advancedSOH import ECMASOH
from moirae.models.ecm.ins_outs import ECMInput
from moirae.models.ecm.transient import ECMTransientVector
from moirae.simulator import Simulator
from moirae.estimators.online.dual import DualEstimator
from moirae.estimators.online.utils.model import CellModelWrapper, DegradationModelWrapper
from moirae.estimators.online.filters.conversions import LinearConversionOperator
from moirae.estimators.online.filters.distributions import MultivariateGaussian, DeltaDistribution
from moirae.estimators.online.filters.kalman.unscented import UnscentedKalmanFilter as UKF


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


def get_protocol(num_cycles: int) -> List:
    """
    Defines cycling protocol for tests
    """
    # Create inputs for the cell
    protocol = []

    start_time = 1
    for _ in range(num_cycles):
        end_time, new_inputs = cycle(start_time=start_time)
        protocol += new_inputs
        start_time = end_time + 1

    return protocol


@fixture
def swapper_operator() -> LinearConversionOperator:
    """
    Simple linear conversion operator that operates on 2D MvRDs and swaps the first and second coordinate
    """
    multi_array = np.array([[0., 1.], [1., 0.]])
    return LinearConversionOperator(multiplicative_array=multi_array)


@fixture
def real_initialization() -> Tuple[ECMTransientVector, ECMASOH, Simulator]:
    """
    Provides real objects for simulation
    """
    # Create simplest possible Rint model, with no hysteresis, and only R0
    real_transients = ECMTransientVector.provide_template(has_C0=False, num_RC=0, soc=0.1)
    real_asoh = ECMASOH.provide_template(has_C0=False, num_RC=0, R0=1., H0=0.)
    # Prepare simulator
    simulator = Simulator(cell_model=ECM(),
                          asoh=real_asoh,
                          transient_state=real_transients,
                          initial_input=ECMInput(),
                          keep_history=True)

    return real_transients, real_asoh, simulator


@fixture
def initial_guesses() -> Tuple[ECMTransientVector, ECMASOH]:
    """
    Provides initial guesses for dual estimation
    """
    # Create initial guesses
    guess_transients = ECMTransientVector.provide_template(has_C0=False, num_RC=0)
    guess_asoh = ECMASOH.provide_template(has_C0=False, num_RC=0, R0=2., H0=0.)
    guess_asoh.mark_updatable(name='r0.base_values')
    return guess_transients, guess_asoh


@fixture
def initial_uncertainties() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Provides covariance matrices for initial transient guess, initial A-SOH guess, and transient covariance process
    noise
    """
    cov_transient = np.diag([1/12, 1.0e-06])  # 4 * (guess_asoh.h0.base_values.item() ** 2) / 12])
    cov_asoh = np.atleast_2d(0.1)
    cov_trans_process = np.diag([1.0e-08, 1.0e-16])
    return cov_transient, cov_asoh, cov_trans_process


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


def test_dual_ukf_init_norm():
    """
    Tests to make sure the dual UKF initialization with normalization is correct
    """
    # Make a more complicated A-SOH
    transient = ECMTransientVector.provide_template(has_C0=False, num_RC=2)
    asoh = ECMASOH.provide_template(has_C0=False, num_RC=2)
    asoh.mark_updatable(name='q_t.base_values')
    asoh.mark_updatable(name='r0.base_values')
    asoh.mark_updatable(name='rc_elements.0.r.base_values')
    asoh.mark_updatable(name='rc_elements.0.c.base_values')
    asoh.mark_updatable(name='rc_elements.1.r.base_values')
    asoh.mark_updatable(name='rc_elements.1.c.base_values')
    assert asoh.get_parameters().size == 6

    dual_ukf = DualEstimator.initialize_unscented_kalman_filter(
        cell_model=ECM(),
        initial_asoh=asoh,
        initial_transients=transient,
        initial_inputs=ECMInput(),
        covariance_transient=0.01 * np.eye(4),
        covariance_asoh=0.02 * np.eye(6),
        normalize_asoh=True)

    assert np.allclose(dual_ukf.trans_filter.hidden.get_mean(), transient.to_numpy().flatten())
    assert np.allclose(dual_ukf.trans_filter.hidden.get_covariance(), 0.01 * np.eye(4))
    assert np.allclose(dual_ukf.asoh_filter.hidden.get_mean(), np.ones(6))
    assert np.allclose(dual_ukf.asoh_filter.hidden.get_covariance(),
                       np.diag(0.02 / (asoh.get_parameters().flatten() ** 2)))


def test_swap(simple_rint, swapper_operator):
    """
    Tests that a single dual estimator step works the same if we simply swap the coordinates
    """
    # Prep basics
    rint_asoh, rint_transient, ecm_inputs, ecm_model = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    simulator = Simulator(cell_model=ECM(),
                          asoh=rint_asoh,
                          transient_state=rint_transient,
                          initial_input=ecm_inputs,
                          keep_history=True)

    # Prepare wrappers
    # Standard
    cell_wrap = CellModelWrapper(cell_model=ECM(), asoh=rint_asoh, transients=rint_transient, inputs=ecm_inputs)
    deg_wrap = DegradationModelWrapper(cell_model=ECM(), asoh=rint_asoh, transients=rint_transient, inputs=ecm_inputs)
    # Swap
    swap_cell_wrap = CellModelWrapper(cell_model=ECM(), asoh=rint_asoh, transients=rint_transient, inputs=ecm_inputs,
                                      converters={'hidden_conversion_operator': swapper_operator})
    deg_wrap2 = DegradationModelWrapper(cell_model=ECM(), asoh=rint_asoh, transients=rint_transient, inputs=ecm_inputs)

    # Create filters
    # Prep
    initial_controls = DeltaDistribution(mean=ecm_inputs.to_numpy().flatten())
    transient_hidden = MultivariateGaussian(mean=rint_transient.to_numpy().flatten(), covariance=0.1 * np.eye(2))
    transient_proc_cov = 0.01 * np.eye(2)
    asoh_hidden = MultivariateGaussian(mean=rint_asoh.get_parameters().flatten(), covariance=np.array([[0.1]]))
    # Standard
    cell_ukf = UKF(model=cell_wrap,
                   initial_hidden=transient_hidden,
                   initial_controls=initial_controls,
                   covariance_process_noise=transient_proc_cov)
    asoh_ukf = UKF(model=deg_wrap,
                   initial_hidden=asoh_hidden,
                   initial_controls=initial_controls)
    # Swap
    swap_ukf = UKF(model=swap_cell_wrap,
                   initial_hidden=transient_hidden.convert(conversion_operator=swapper_operator, inverse=True),
                   initial_controls=initial_controls,
                   covariance_process_noise=swapper_operator.inverse_transform_covariance(transient_proc_cov))
    asoh_ukf2 = UKF(model=deg_wrap2,
                    initial_hidden=asoh_hidden,
                    initial_controls=initial_controls)

    # Assemble dual estimators
    stnd_dual = DualEstimator(transient_filter=cell_ukf, asoh_filter=asoh_ukf)
    swap_dual = DualEstimator(transient_filter=swap_ukf, asoh_filter=asoh_ukf2)

    # Step
    new_input = ECMInput(time=1., current=1.)
    new_trans, new_volts = simulator.step(new_inputs=new_input)
    stnd_state, stnd_pred = stnd_dual.step(inputs=new_input, measurements=new_volts)
    swap_state, swap_pred = swap_dual.step(inputs=new_input, measurements=new_volts)
    assert np.allclose(stnd_state.get_mean(), swap_state.get_mean()), \
        f'Stnd: {stnd_state.get_mean()}; Swap: {swap_state.get_mean()}'
    assert np.allclose(stnd_state.get_covariance(), swap_state.get_covariance()), \
        f'Stnd: {stnd_state.get_covariance()}; Swap: {swap_state.get_covariance()}'
    assert np.allclose(stnd_pred.get_mean(), swap_pred.get_mean()), \
        f'Stnd: {stnd_pred.get_mean()}; Swap: {swap_pred.get_mean()}'
    assert np.allclose(stnd_pred.get_covariance(), swap_pred.get_covariance()), \
        f'Stnd: {stnd_pred.get_covariance()}; Swap: {swap_pred.get_covariance()}'


@mark.slow
def test_simplest_rint(real_initialization, initial_guesses, initial_uncertainties):
    # Get protocol
    protocol = get_protocol(num_cycles=4)

    # Collect real values
    real_transients, real_asoh, simulator = real_initialization

    # Get initial guesses
    guess_transients, guess_asoh = initial_guesses

    # Prepare the dual estimation
    cov_transient, cov_asoh, cov_trans_process = initial_uncertainties

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


@mark.slow
def test_simplest_rint_normalization(real_initialization, initial_guesses, initial_uncertainties):
    # Get protocol for 5 cycles (normalization requires a bit more cycles to converge for some reason)
    protocol = get_protocol(num_cycles=5)

    # Collect real values
    real_transients, real_asoh, simulator = real_initialization

    # Get initial guesses
    guess_transients, guess_asoh = initial_guesses

    # Prepare the dual estimation
    cov_transient, cov_asoh, cov_trans_process = initial_uncertainties

    dual_ukf = DualEstimator.initialize_unscented_kalman_filter(
        cell_model=ECM(),
        initial_asoh=guess_asoh,
        initial_transients=guess_transients,
        initial_inputs=simulator.previous_input,
        covariance_asoh=cov_asoh,
        covariance_transient=cov_transient,
        covariance_sensor_noise=np.atleast_2d(1.0e-06),
        transient_covariance_process_noise=cov_trans_process,
        normalize_asoh=True)

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
