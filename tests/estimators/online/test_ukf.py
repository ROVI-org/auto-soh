import numpy as np

from moirae.estimators.online import ControlVariables, OutputMeasurements
from moirae.estimators.online.kalman.unscented import UnscentedKalmanFilter as UKF
from moirae.estimators.online.kalman import KalmanHiddenState
from moirae.models.base import CellModel, InputQuantities, GeneralContainer, HealthVariable, OutputQuantities


# Define Lorenz dynamics

class LorenzState(GeneralContainer):
    x: float
    y: float
    z: float


class LorenzControl(InputQuantities):
    sigma: float
    rho: float
    beta: float
    n: float


class LorenzOutputs(OutputQuantities):
    m1: float


class LorenzModel(CellModel):

    def update_transient_state(
            self,
            previous_input: LorenzControl,
            current_input: LorenzControl,
            transient_state: LorenzState,
            asoh: HealthVariable
    ) -> GeneralContainer:
        dt = current_input.time - previous_input.time

        # Compute derivatives
        dxdt = current_input.sigma * (transient_state.y - transient_state.x)
        dydt = transient_state.x * (current_input.rho - transient_state.z) - transient_state.y
        dzdt = (transient_state.x * transient_state.y) - current_input.beta * transient_state.z

        return LorenzState(
            x=dt * dxdt,
            y=dt * dydt,
            z=dt * dzdt,
        )

    def calculate_terminal_voltage(
            self,
            inputs: LorenzControl,
            transient_state: LorenzState,
            asoh: HealthVariable) -> OutputQuantities:
        hidden_states = transient_state.to_numpy()

        return LorenzOutputs(
            terminal_voltage=np.sqrt(np.sum(hidden_states ** 2)),
            m1=abs(np.sum(hidden_states ** inputs.n)) ** (1. / inputs.n),
        )


def test_lorenz_ukf():
    # Initiate RNG
    rng = np.random.default_rng(314159)

    # Approximate initial state
    state0 = LorenzState(x=0., y=0., z=0.)
    cov_state = np.diag([20, 30, 10])

    # Noise terms
    process_noise = 1.0e-02 * np.eye(3)
    sensor_noise = np.eye(2)

    # Initial control
    u0 = LorenzControl(
        time=0.,
        current=0.,  # Not used by Lorenz
        sigma=10.,
        rho=28.,
        beta=8. / 3.,
        n=2
    )

    # Define UKF
    initial_state = KalmanHiddenState(
        mean=state0.to_numpy() + rng.multivariate_normal(mean=np.zeros(3), cov=cov_state),
        covariance=cov_state
    )
    ukf_chaos = UKF(model=LorenzModel(),
                    initial_asoh=HealthVariable(),  # No SOH for lorenz
                    initial_transients=state0,
                    initial_inputs=u0,
                    initial_covariance=cov_state,
                    covariance_process_noise=process_noise,
                    covariance_sensor_noise=sensor_noise)

    # Initialize dictionaries to store values
    real_values = {'state': [np.array([state0])],
                   'measurements': []}
    noisy_values = {'state': [initial_state.get_mean()],
                    'measurements': []}
    ukf_values = {'state': [initial_state.model_copy(deep=True)],
                  'measurements': []}

    # Timestamps
    timestamps = [0.0]

    # Assign previous control
    previous_control = ControlVariables(mean=u0.to_numpy())

    for _ in range(10000):
        # Get a new time
        time = timestamps[-1] + 0.01 * (1. + rng.random())
        timestamps += [time]
        # Choose random new controls
        sigma = 10
        beta = 8. / 3.
        rho = rng.normal(loc=28, scale=4)
        n = rng.integers(2, 5)
        u = ControlVariables(mean=np.array([time, 0., sigma, rho, beta, n]))

        # Compute new true hidden state
        prev_hidden = noisy_values['state'][-1]
        new_state = ukf_chaos.update_hidden_states(hidden_states=prev_hidden[None, :],
                                                   previous_controls=previous_control,
                                                   new_controls=u)[0, :]
        real_values['state'] += [new_state.copy()]
        new_state += rng.multivariate_normal(mean=np.zeros(3), cov=process_noise)
        noisy_values['state'] += [new_state.copy()]

        # Get new measurement
        m = ukf_chaos.predict_measurement(hidden_states=new_state[None, :], controls=u)
        real_values['measurements'] += [m.copy()]
        m += rng.multivariate_normal(mean=np.zeros(2), cov=sensor_noise)
        noisy_values['measurements'] += [m.copy()]

        # Assemble measurement
        measure = OutputMeasurements(mean=m)
        ukf_pred, ukf_hid = ukf_chaos.step(u=u, y=measure)
        ukf_values['state'].append(ukf_hid)
        ukf_values['measurements'].append(ukf_pred)

        # Update previous control
        previous_control = u

    # Getting relevant values
    ukf_m0 = np.array([y.mean[0] for y in ukf_values['measurements']])
    ukf_m1 = np.array([y.mean[1] for y in ukf_values['measurements']])
    ukf_m0_std = np.sqrt(np.array([y.covariance[0, 0] for y in ukf_values['measurements']]))
    ukf_m1_std = np.sqrt(np.array([y.covariance[1, 1] for y in ukf_values['measurements']]))
    ukf_x = np.array([hid.mean[0] for hid in ukf_values['state']])
    ukf_y = np.array([hid.mean[1] for hid in ukf_values['state']])
    ukf_z = np.array([hid.mean[2] for hid in ukf_values['state']])
    ukf_x_std = np.sqrt(np.array([hid.covariance[0, 0] for hid in ukf_values['state']]))
    ukf_y_std = np.sqrt(np.array([hid.covariance[1, 1] for hid in ukf_values['state']]))
    ukf_z_std = np.sqrt(np.array([hid.covariance[2, 2] for hid in ukf_values['state']]))

    # Get the noisy measurements
    m0_noise = np.array([y.flatten()[0] for y in noisy_values['measurements']])
    m1_noise = np.array([y.flatten()[1] for y in noisy_values['measurements']])

    # Check against UKF with error
    m0_1sig = np.isclose(m0_noise, ukf_m0, atol=1 * ukf_m0_std)
    m0_2sig = np.isclose(m0_noise, ukf_m0, atol=2 * ukf_m0_std)
    m0_3sig = np.isclose(m0_noise, ukf_m0, atol=3 * ukf_m0_std)
    m1_1sig = np.isclose(m1_noise, ukf_m1, atol=1 * ukf_m1_std)
    m1_2sig = np.isclose(m1_noise, ukf_m1, atol=2 * ukf_m1_std)
    m1_3sig = np.isclose(m1_noise, ukf_m1, atol=3 * ukf_m1_std)

    # Only consider the last few points, when the UKF should have converged
    last_points = 500
    m0_1sig_frac = np.sum(m0_1sig[-last_points:]) / last_points
    m0_2sig_frac = np.sum(m0_2sig[-last_points:]) / last_points
    m0_3sig_frac = np.sum(m0_3sig[-last_points:]) / last_points
    m1_1sig_frac = np.sum(m1_1sig[-last_points:]) / last_points
    m1_2sig_frac = np.sum(m1_2sig[-last_points:]) / last_points
    m1_3sig_frac = np.sum(m1_3sig[-last_points:]) / last_points

    # Check stats, but be slightly more lenient than a true Gaussian
    assert m0_1sig_frac >= 0.625, 'Fraction within 1 standard deviation is %2.1f %% < 62.5%%!!' % (
            100 * m0_1sig_frac)
    assert m0_2sig_frac >= 0.925, 'Fraction within 2 standard deviations is %2.1f %% < 92.5%%!!' % (
            100 * m0_2sig_frac)
    assert m0_3sig_frac >= 0.975, 'Fraction within 3 standard deviations is %2.1f %% < 97.5%%!!' % (
            100 * m0_3sig_frac)
    assert m1_1sig_frac >= 0.625, 'Fraction within 1 standard deviation is %2.1f %% < 62.5%%!!' % (
            100 * m1_1sig_frac)
    assert m1_2sig_frac >= 0.925, 'Fraction within 2 standard deviations is %2.1f < 92.5%%!!' % (100 * m1_2sig_frac)
    assert m1_3sig_frac >= 0.975, 'Fraction within 3 standard deviations is %2.1f %% < 97.5%%!!' % (
            100 * m1_3sig_frac)

    # Getting noisy hidden states
    x_noise = np.array([hid.flatten()[0] for hid in noisy_values['state']])
    y_noise = np.array([hid.flatten()[1] for hid in noisy_values['state']])
    z_noise = np.array([hid.flatten()[2] for hid in noisy_values['state']])

    # Check against error
    x_1sig = np.isclose(x_noise, ukf_x, atol=1 * ukf_x_std)
    x_2sig = np.isclose(x_noise, ukf_x, atol=2 * ukf_x_std)
    x_3sig = np.isclose(x_noise, ukf_x, atol=3 * ukf_x_std)
    y_1sig = np.isclose(y_noise, ukf_y, atol=1 * ukf_y_std)
    y_2sig = np.isclose(y_noise, ukf_y, atol=2 * ukf_y_std)
    y_3sig = np.isclose(y_noise, ukf_y, atol=3 * ukf_y_std)
    z_1sig = np.isclose(z_noise, ukf_z, atol=1 * ukf_z_std)
    z_2sig = np.isclose(z_noise, ukf_z, atol=2 * ukf_z_std)
    z_3sig = np.isclose(z_noise, ukf_z, atol=3 * ukf_z_std)

    # Only consider the last few points again
    x_1sig_frac = np.sum(x_1sig[-last_points:]) / last_points
    x_2sig_frac = np.sum(x_2sig[-last_points:]) / last_points
    x_3sig_frac = np.sum(x_3sig[-last_points:]) / last_points
    y_1sig_frac = np.sum(y_1sig[-last_points:]) / last_points
    y_2sig_frac = np.sum(y_2sig[-last_points:]) / last_points
    y_3sig_frac = np.sum(y_3sig[-last_points:]) / last_points
    z_1sig_frac = np.sum(z_1sig[-last_points:]) / last_points
    z_2sig_frac = np.sum(z_2sig[-last_points:]) / last_points
    z_3sig_frac = np.sum(z_3sig[-last_points:]) / last_points

    # Check stats, but give a wider margin than a true Gaussian, as this is a very chaotic system and we are looking at
    # the hidden states, instead of number the UKF can measure itself against.
    assert x_1sig_frac > 0.60, 'Fraction within 1 standard deviation is %2.1f %% < 60%%!!' % (100 * x_1sig_frac)
    assert x_2sig_frac > 0.90, 'Fraction within 1 standard deviation is %2.1f %% < 90%%!!' % (100 * x_2sig_frac)
    assert x_3sig_frac > 0.95, 'Fraction within 1 standard deviation is %2.1f %% < 95%%!!' % (100 * x_3sig_frac)
    assert y_1sig_frac > 0.60, 'Fraction within 1 standard deviation is %2.1f %% < 60%%!!' % (100 * y_1sig_frac)
    assert y_2sig_frac > 0.90, 'Fraction within 1 standard deviation is %2.1f %% < 90%%!!' % (100 * y_2sig_frac)
    assert y_3sig_frac > 0.95, 'Fraction within 1 standard deviation is %2.1f %% < 95%%!!' % (100 * y_3sig_frac)
    assert z_1sig_frac > 0.60, 'Fraction within 1 standard deviation is %2.1f %% < 60%%!!' % (100 * z_1sig_frac)
    assert z_2sig_frac > 0.90, 'Fraction within 1 standard deviation is %2.1f %% < 90%%!!' % (100 * z_2sig_frac)
    assert z_3sig_frac > 0.95, 'Fraction within 1 standard deviation is %2.1f %% < 95%%!!' % (100 * z_3sig_frac)
