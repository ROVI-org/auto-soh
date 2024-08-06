from typing import List

from scipy.linalg import block_diag
import numpy as np

from moirae.estimators.online.distributions import DeltaDistribution
from moirae.models.ecm import EquivalentCircuitModel as ECM
from moirae.models.ecm.advancedSOH import ECMASOH
from moirae.models.ecm.ins_outs import ECMInput, ECMMeasurement
from moirae.models.ecm.transient import ECMTransientVector
from moirae.simulator import Simulator
from moirae.estimators.online.kalman.unscented import JointUnscentedKalmanFilter as JointUKF


def cycle_protocol(rng, asoh: ECMASOH, start_time: float = 0.0) -> List[ECMInput]:
    """
    Function to return list of ECMInputs corresponding to a new cycle

    Args:
        asoh: A-SOH object to be used
        start_time: time at which to start this cycle in seconds

    Returns:
        ecm_inputs: list of ECMInput objects corresponding to this cycle
    """
    # initialize main lists to be used
    timestamps = []
    currents = []

    # Get the Qt and CE, which will help determine the currents
    qt = asoh.q_t.amp_hour
    ce = asoh.ce.item()

    # Now, let's choose rest durations and C-rates
    rest0, rest1, rest2 = rng.integers(low=5, high=10, size=3)
    rest0 *= 60
    rest1 *= 2 * 60
    rest2 *= 60
    dischg_rate, chg_rate = np.array([1.5, 0.5]) + rng.random(size=2)

    def update_timestamps_curr(duration: float, curr: float) -> None:
        """
        Helper to populate timestamps and currents.

        Args:
            duration: duration of segment in seconds
            curr: value of current to be used.
        """
        # specify binds to nearest variables
        nonlocal currents
        nonlocal timestamps
        new_len = int(np.ceil(duration))
        currents += [curr] * new_len
        new_times = np.sort(rng.random(size=new_len)) * duration
        if len(timestamps) == 0:
            new_times += start_time
        else:
            new_times += timestamps[-1] + (1.0e-03)
        timestamps += new_times.tolist()

    # Rest 0
    update_timestamps_curr(duration=rest0, curr=0.0)

    # Discharge
    dischg_duration = 3600.0 / dischg_rate
    dischg_curr = -(qt * dischg_rate)
    update_timestamps_curr(duration=dischg_duration, curr=dischg_curr)

    # Rest 1
    update_timestamps_curr(duration=rest1, curr=0.0)

    # Charge
    chg_duration = 3600.0 / chg_rate
    chg_curr = (qt * chg_rate) / ce
    update_timestamps_curr(duration=chg_duration, curr=chg_curr)

    # Rest 2
    update_timestamps_curr(duration=rest2, curr=0.0)

    # Finally, assemble the ECMInputs
    ecm_inputs = [ECMInput(time=time, current=current) for time, current in zip(timestamps, currents)]
    return ecm_inputs


def test_normalization(simple_rint):
    """Make sure that normalizing the ASOH parameters works as expected"""

    # Make the serial resistor updatable
    rint_asoh, rint_transient, ecm_inputs, ecm_model = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    assert rint_asoh.num_updatable == 1

    # Make the filter without normalization
    init_covar = np.diag([0.1, 0.1, 0.1])
    r0 = rint_asoh.r0.base_values.item()
    ukf_joint = JointUKF(
        model=ecm_model,
        initial_asoh=rint_asoh,
        initial_transients=rint_transient,
        initial_inputs=ecm_inputs,
        initial_covariance=np.diag([0.1, 0.1, 0.1]),
        normalize_asoh=False
    )
    assert ukf_joint.num_transients == 2  # SOC, hysteresis
    assert ukf_joint.num_hidden_dimensions == 3  # Includes r_int
    assert ukf_joint.num_output_dimensions == 1  # Just the Voltage
    assert np.allclose(ukf_joint.joint_normalization_factor, 1.)
    assert np.allclose(ukf_joint.covariance_normalization, 1.)
    assert np.allclose(ukf_joint.state.covariance, np.diag([0.1] * 3))
    assert np.allclose(ukf_joint.state.mean, np.concatenate([rint_transient.to_numpy()[0, :], [r0]]))

    # Make the filter with normalization
    ukf_joint_normed = JointUKF(
        model=ecm_model,
        initial_asoh=rint_asoh,
        initial_transients=rint_transient,
        initial_inputs=ecm_inputs,
        initial_covariance=init_covar,
        normalize_asoh=True
    )
    assert np.allclose(ukf_joint_normed.joint_normalization_factor, [1., 1., r0])
    assert np.allclose(ukf_joint_normed.covariance_normalization,
                       [[1., 1., r0], [1., 1., r0], [r0, r0, r0 ** 2]])
    assert np.allclose(ukf_joint_normed.state.covariance, np.diag([0.1, 0.1, 0.1 / r0 ** 2]))
    assert np.allclose(ukf_joint_normed.state.mean, np.concatenate([rint_transient.to_numpy()[0, :], [1]]))

    # Make sure the two filters step correctly
    applied_control = DeltaDistribution(mean=np.array([1., 1., 25.]))  # Time=1s, I=1 Amp, 25C
    end_soc = (1. / rint_asoh.q_t.value).item()
    end_voltage = (rint_asoh.ocv.get_value(soc=end_soc) + 1. * rint_asoh.r0.get_value(soc=end_soc)).item()
    observed_voltage = DeltaDistribution(mean=np.array([end_voltage]))
    for ukf in [ukf_joint, ukf_joint_normed]:
        # Test that it runs the cell model properly
        updated_states = ukf.update_hidden_states(
            hidden_states=ukf.state.mean[None, :],
            previous_controls=DeltaDistribution(mean=np.array([0, 1])),
            new_controls=applied_control
        )
        actual_states = ukf._denormalize_hidden_array(updated_states)
        assert np.allclose(actual_states, [[end_soc, 0., r0]])

        # Test that it gets the outputs correctly
        pred_outputs = ukf.predict_measurement(updated_states, controls=applied_control)
        assert np.allclose(pred_outputs, end_voltage)

        # Make sure nothing odd happens
        pred_voltage, pred_state = ukf.step(applied_control, observed_voltage)
        assert np.isfinite(pred_voltage.get_mean()).all()
        assert np.isfinite(pred_state.get_mean()).all()


def test_names(simple_rint):
    rint_asoh, rint_transient, ecm_inputs, ecm_model = simple_rint
    rint_asoh.mark_updatable('r0.base_values')
    ukf_joint = JointUKF(
        model=ecm_model,
        initial_asoh=rint_asoh,
        initial_transients=rint_transient,
        initial_inputs=ecm_inputs,
        normalize_asoh=False
    )
    assert ukf_joint.state_names == ('soc', 'hyst', 'r0.base_values')
    assert ukf_joint.output_names == ('terminal_voltage',)
    assert ukf_joint.control_names == ('time', 'current', 'temperature')


def test_joint_ecm() -> None:
    # Initialize RNG
    rng = np.random.default_rng(seed=31415926535897932384626433832)

    # Initialize A-SOH and make Qt and R0 updatable
    asoh_rint = ECMASOH.provide_template(has_C0=False, num_RC=0)
    asoh_rint.mark_updatable(name='q_t.base_values')
    asoh_rint.mark_updatable(name='r0.base_values')

    # Initialize transient state and simulator
    transient0_rint = ECMTransientVector.provide_template(has_C0=False, num_RC=0, soc=1.0)
    rint_sim = Simulator(
        model=ECM(),
        asoh=asoh_rint,
        initial_input=ECMInput(),
        transient_state=transient0_rint,
        keep_history=True
    )

    # Set up covariances for estimator
    # Start with the A-SOH
    cov_asoh_rint = [6.25e-04]  # Qt: +/- 0.05 Amp-hour
    cov_asoh_rint += [2.5e-05]  # R0: +/- 10 mOhm
    cov_asoh_rint = np.diag(cov_asoh_rint)

    # Now, the transient
    cov_tran_rint = [1. / 12]  # SOC
    cov_tran_rint += [4 * (asoh_rint.h0.base_values.item()) ** 2 / 12]  # hyst
    cov_tran_rint = np.diag(cov_tran_rint)

    # Generate perturbed values
    tran_rint_perturb = rng.multivariate_normal(mean=transient0_rint.to_numpy()[0, :], cov=cov_tran_rint)
    asoh_rint_perturb = rng.multivariate_normal(mean=asoh_rint.get_parameters()[0, :], cov=cov_asoh_rint)
    # Make sure the initial SOC is not too off-base
    while tran_rint_perturb[0] > 1.05:
        tran_rint_perturb[0] = rng.normal(loc=1.0, scale=cov_tran_rint[0, 0])

    # Finally, prepare the objects for the joint estimation
    asoh_rint_off = asoh_rint.model_copy(deep=True)
    transient0_rint_off = transient0_rint.model_copy(deep=True)
    asoh_rint_off.update_parameters(asoh_rint_perturb)
    transient0_rint_off.from_numpy(tran_rint_perturb)

    # Create joint estimator by setting very small noise terms for the A-SOH parameters
    voltage_err = 1.0e-03  # mV voltage error
    noise_sensor = ((voltage_err / 2) ** 2) * np.eye(1)
    noise_asoh = 1.0e-10 * np.eye(len(asoh_rint_perturb))
    noise_asoh[0, 0] = 6.25e-04  # Qt value is 10 Amp-hour, so +/- 0.05 Amp-hour is reasonable
    noise_tran = 1.0e-08 * np.eye(2)
    rint_joint_ukf = JointUKF(
        model=ECM(),
        initial_transients=transient0_rint_off,
        initial_asoh=asoh_rint_off,
        initial_inputs=rint_sim.previous_input.model_copy(deep=True),
        initial_covariance=block_diag(cov_tran_rint, cov_asoh_rint),
        # TODO (wardlt): Consider keeping ASOH and transients as separate kwargs
        covariance_process_noise=block_diag(noise_tran, noise_asoh),
        covariance_sensor_noise=noise_sensor,
        normalize_asoh=False
    )

    # Co-simulate
    # Total number of cycles to simulate
    num_cycles = 10

    # Let's also create a simple ways to store UKF predictions and estimates, as well as the inputs provided
    noisy_voltage = []
    controls = []
    joint_ukf_predictions = {'joint_states': [], 'voltages': []}

    # Specify the start time of the simulation
    start_time = 0.0

    for _ in range(num_cycles):
        # Generate list of inputs and store them in the controls
        protocol = cycle_protocol(rng, asoh=rint_sim.asoh, start_time=start_time)
        controls += protocol

        for new_input in protocol:
            # Simulate (useful for also getting the real terminal voltage :) )
            _, cell_response = rint_sim.step(new_inputs=new_input)
            # Add noise to give to the UKF and store it
            vt = cell_response.terminal_voltage + rng.normal(loc=0.0, scale=voltage_err / 2)
            noisy_voltage += [vt]
            # Step the joint estimator
            measurement = ECMMeasurement(terminal_voltage=vt)
            pred_measure, est_hidden = rint_joint_ukf.step(
                u=DeltaDistribution(mean=new_input.to_numpy()),
                y=DeltaDistribution(mean=measurement.to_numpy())
            )
            # Save to the dictionary
            joint_ukf_predictions['joint_states'] += [est_hidden.model_copy(deep=True)]
            joint_ukf_predictions['voltages'] += [pred_measure.model_copy(deep=True)]

        # Update the start time of the next cycle
        start_time = protocol[-1].time.item() + 1.0e-03

    # Collect results for validation
    # Get real values
    real_soc = np.array([transient.soc.item() for transient in rint_sim.transient_history])
    real_hyst = np.array([transient.hyst.item() for transient in rint_sim.transient_history])

    # Estimate values and their corresponding uncertainty
    estimated_soc = np.array([estimate.mean[0] for estimate in joint_ukf_predictions['joint_states']])
    estimated_hyst = np.array([estimate.mean[1] for estimate in joint_ukf_predictions['joint_states']])
    estimated_Qt = np.array([estimate.mean[2] for estimate in joint_ukf_predictions['joint_states']])
    estimated_R0 = np.array([estimate.mean[3] for estimate in joint_ukf_predictions['joint_states']])
    estimated_soc_std = np.sqrt(np.array([estimate.covariance[0, 0]
                                          for estimate in joint_ukf_predictions['joint_states']]))
    estimated_hyst_std = np.sqrt(np.array([estimate.covariance[1, 1]
                                           for estimate in joint_ukf_predictions['joint_states']]))
    estimated_Qt_std = np.sqrt(np.array([estimate.covariance[2, 2]
                                         for estimate in joint_ukf_predictions['joint_states']]))
    estimated_R0_std = np.sqrt(np.array([estimate.covariance[3, 3]
                                         for estimate in joint_ukf_predictions['joint_states']]))
    predicted_Vt = np.array([prediction.mean[0] for prediction in joint_ukf_predictions['voltages']])
    predicted_Vt_std = np.sqrt(np.array([prediction.covariance[0, 0]
                                         for prediction in joint_ukf_predictions['voltages']]))

    # Check statistics
    # Number of iterations to consider
    num_iterations = 5 * 3600  # ~5 hours

    # Predictions
    soc_sigs = []
    hyst_sigs = []
    qt_sigs = []
    r0_sigs = []
    vt_sigs = []
    for i in range(1, 4):
        soc_sigs.append(np.sum(
            np.isclose(real_soc[-num_iterations:],
                       estimated_soc[-num_iterations:],
                       atol=(i * estimated_soc_std[-num_iterations:]))) / num_iterations)
        hyst_sigs.append(np.sum(
            np.isclose(real_hyst[-num_iterations:],
                       estimated_hyst[-num_iterations:],
                       atol=(i * estimated_hyst_std[-num_iterations:]))) / num_iterations)
        qt_sigs.append(np.sum(
            np.isclose(asoh_rint.q_t.base_values,
                       estimated_Qt[-num_iterations:],
                       atol=(i * estimated_Qt_std[-num_iterations:]))) / num_iterations)
        r0_sigs.append(np.sum(
            np.isclose(asoh_rint.r0.base_values,
                       estimated_R0[-num_iterations:],
                       atol=(i * estimated_R0_std[-num_iterations:]))) / num_iterations)
        vt_sigs.append(np.sum(
            np.isclose(noisy_voltage[-num_iterations:],
                       predicted_Vt[-num_iterations:],
                       atol=(i * predicted_Vt_std[-num_iterations:]))) / num_iterations)

    # Metrics
    gaussian_metrics = {1: 68., 2: 95., 3: 98.}

    # Check values within 2 and 3 srd
    for i in [2, 3]:
        assert soc_sigs[i - 1] >= gaussian_metrics[i] / 100, \
            'Real SOC value is within %d std of prediction only %2.2f%% of the time, rather than %2.0f%%!' % \
            (i, 100 * soc_sigs[i - 1], gaussian_metrics[i])
        assert hyst_sigs[i - 1] >= gaussian_metrics[i] / 100, \
            'Real hysteresis value is within %d std of prediction only %2.2f%% of the time, rather than %2.0f%%!' % \
            (i, 100 * hyst_sigs[i - 1], gaussian_metrics[i])
        assert qt_sigs[i - 1] >= gaussian_metrics[i] / 100, \
            'Real Qt value is within %d std of prediction only %2.2f%% of the time, rather than %2.0f%%!' % \
            (i, 100 * qt_sigs[i - 1], gaussian_metrics[i])
        assert r0_sigs[i - 1] >= gaussian_metrics[i] / 100, \
            'Real R0 value is within %d std of prediction only %2.2f%% of the time, rather than %2.0f%%!' % \
            (i, 100 * r0_sigs[i - 1], gaussian_metrics[i])
        assert vt_sigs[i - 1] >= gaussian_metrics[i] / 100, \
            'Real Vt value is within %d std of prediction only %2.2f%% of the time, rather than %2.0f%%!' % \
            (i, 100 * qt_sigs[i - 1], gaussian_metrics[i])
