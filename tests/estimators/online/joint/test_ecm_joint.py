from typing import List

import numpy as np
import pytest

from asoh.models.ecm.advancedSOH import ECMASOH
from asoh.models.ecm.ins_outs import ECMInput, ECMMeasurement
from asoh.models.ecm.transient import ECMTransientVector
from asoh.models.ecm.simulator import ECMSimulator
from asoh.estimators.online.joint.ecm.unscented import ECMJointUKF


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
    ce = asoh.ce

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


@pytest.mark.slow
def test_joint_ecm() -> None:
    # Initialize RNG
    rng = np.random.default_rng(seed=31415926535897932384626433832)

    # Initialize A-SOH and make Qt and R0 updatable
    asoh_rint = ECMASOH.provide_template(has_C0=False, num_RC=0)
    asoh_rint.mark_updatable(name='q_t.base_values')
    asoh_rint.mark_updatable(name='r0.base_values')

    # Initialize transient state and simulator
    transient0_rint = ECMTransientVector.provide_template(has_C0=False, num_RC=0, soc=1.0)
    rint_sim = ECMSimulator(asoh=asoh_rint, transient_state=transient0_rint, keep_history=True)

    # Set up covariances for estimator
    # Start with the A-SOH
    cov_asoh_rint = [6.25e-04]  # Qt: +/- 0.05 Amp-hour
    cov_asoh_rint += [2.5e-05]  # R0: +/- 10 mOhm
    cov_asoh_rint = np.diag(cov_asoh_rint)

    # Now, the transient
    cov_tran_rint = [1./12]  # SOC
    cov_tran_rint += [4 * (asoh_rint.h0.base_values)**2 / 12]  # hyst
    cov_tran_rint = np.diag(cov_tran_rint)

    # Generate perturbed values
    tran_rint_perturb = rng.multivariate_normal(mean=transient0_rint.to_numpy(), cov=cov_tran_rint)
    asoh_rint_perturb = rng.multivariate_normal(mean=asoh_rint.get_parameters(), cov=cov_asoh_rint)
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
    rint_joint_ukf = ECMJointUKF(initial_transient=transient0_rint_off,
                                 initial_asoh=asoh_rint_off,
                                 initial_control=rint_sim.previous_input.model_copy(deep=True),
                                 covariance_transient=cov_tran_rint,
                                 covariance_asoh=cov_asoh_rint,
                                 transient_noise=noise_tran,
                                 asoh_noise=noise_asoh,
                                 sensor_noise=noise_sensor,
                                 normalize_asoh=True)

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
            _, cell_response = rint_sim.step(new_input=new_input)
            # Add noise to give to the UKF and store it
            vt = cell_response.terminal_voltage + rng.normal(loc=0.0, scale=voltage_err / 2)
            noisy_voltage += [vt]
            # Step the joint estimator``
            measurement = ECMMeasurement(terminal_voltage=vt)
            pred_measure, est_hidden = rint_joint_ukf.step(u=new_input, y=measurement)
            # Save to the dictionary
            joint_ukf_predictions['joint_states'] += [est_hidden.model_copy(deep=True)]
            joint_ukf_predictions['voltages'] += [pred_measure.model_copy(deep=True)]

        # Update the start time of the next cycle
        start_time = protocol[-1].time + 1.0e-03

    # Collect results for validation
    # Get real values
    real_soc = np.array([transient.soc for transient in rint_sim.transient_history])
    real_hyst = np.array([transient.hyst for transient in rint_sim.transient_history])

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
    num_iterations = 5 * 3600   # ~5 hours

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
                       atol=(i*estimated_soc_std[-num_iterations:]))) / num_iterations)
        hyst_sigs.append(np.sum(
            np.isclose(real_hyst[-num_iterations:],
                       estimated_hyst[-num_iterations:],
                       atol=(i*estimated_hyst_std[-num_iterations:]))) / num_iterations)
        qt_sigs.append(np.sum(
            np.isclose(asoh_rint.q_t.base_values,
                       estimated_Qt[-num_iterations:],
                       atol=(i*estimated_Qt_std[-num_iterations:]))) / num_iterations)
        r0_sigs.append(np.sum(
            np.isclose(asoh_rint.r0.base_values,
                       estimated_R0[-num_iterations:],
                       atol=(i*estimated_R0_std[-num_iterations:]))) / num_iterations)
        vt_sigs.append(np.sum(
            np.isclose(noisy_voltage[-num_iterations:],
                       predicted_Vt[-num_iterations:],
                       atol=(i*predicted_Vt_std[-num_iterations:]))) / num_iterations)

    # Metrics
    gaussian_metrics = {1: 68., 2: 95., 3: 98.}

    # Check values within 2 and 3 srd
    for i in [2, 3]:
        assert soc_sigs[i-1] >= gaussian_metrics[i] / 100, \
            'Real SOC value is within %d std of prediction only %2.2f%% of the time, rather than %2.0f%%!' % \
            (i, 100 * soc_sigs[i-1], gaussian_metrics[i])
        assert hyst_sigs[i-1] >= gaussian_metrics[i] / 100, \
            'Real hysteresis value is within %d std of prediction only %2.2f%% of the time, rather than %2.0f%%!' % \
            (i, 100 * hyst_sigs[i-1], gaussian_metrics[i])
        assert qt_sigs[i-1] >= gaussian_metrics[i] / 100, \
            'Real Qt value is within %d std of prediction only %2.2f%% of the time, rather than %2.0f%%!' % \
            (i, 100 * qt_sigs[i-1], gaussian_metrics[i])
        assert r0_sigs[i-1] >= gaussian_metrics[i] / 100, \
            'Real R0 value is within %d std of prediction only %2.2f%% of the time, rather than %2.0f%%!' % \
            (i, 100 * r0_sigs[i-1], gaussian_metrics[i])
        assert vt_sigs[i-1] >= gaussian_metrics[i] / 100, \
            'Real Vt value is within %d std of prediction only %2.2f%% of the time, rather than %2.0f%%!' % \
            (i, 100 * qt_sigs[i-1], gaussian_metrics[i])
