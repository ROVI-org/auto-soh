import numpy as np
from pytest import fixture

from asoh.estimators.online import ModelFilterInterface, ControlVariables, OutputMeasurements
from asoh.estimators.online.general.kalman.unscented import UnscentedKalmanFilter as UKF
from asoh.estimators.online.general.kalman import KalmanHiddenState


# Define Lorenz dynamics
class Lorenz(ModelFilterInterface):
    def __init__(self) -> None:
        pass

    @property
    def num_hidden_dimensions(self) -> int:
        """ Outputs expected dimensionality of hidden state """
        return 3

    @property
    def num_output_dimensions(self) -> int:
        """ Outputs expected dimensionality of output measurements """
        return 2

    def update_hidden_states(self,
                             hidden_states: np.ndarray,
                             previous_controls: ControlVariables,
                             new_controls: ControlVariables) -> np.ndarray:
        # Get attractor params
        sigma = new_controls.mean[1]
        rho = new_controls.mean[2]
        beta = new_controls.mean[3]
        dt = new_controls.mean[0] - previous_controls.mean[0]

        # Compute derivatives
        dxdt = sigma * (hidden_states[:, 1] - hidden_states[:, 0])
        dydt = hidden_states[:, 0] * (rho - hidden_states[:, 2]) - hidden_states[:, 1]
        dzdt = (hidden_states[:, 0] * hidden_states[:, 1]) - beta * hidden_states[:, 2]

        # Build matrix
        update_mat = dt * np.hstack((dxdt.reshape((-1, 1)), dydt.reshape((-1, 1)), dzdt.reshape((-1, 1))))

        return hidden_states + update_mat

    def predict_measurement(self,
                            hidden_states: np.ndarray,
                            controls: ControlVariables) -> np.ndarray:
        # Get power parameter
        n = controls.mean[4]

        # Compute outputs
        m0 = ((np.sum(hidden_states ** 2, axis=1)) ** (1./2.)).reshape((-1, 1))
        m1 = (abs(np.sum(hidden_states ** n, axis=1)) ** (1./n)).reshape((-1, 1))
        return np.hstack((m0, m1))


@fixture
def lorenz_model() -> Lorenz:
    return Lorenz()


def test_lorenz_ukf(lorenz_model):
    # Initiate RNG
    rng = np.random.default_rng(314159)

    # Approximate initial state
    state0 = np.array([0., 0., 0.])
    cov_state = np.diag([20, 30, 10])

    # Noise terms
    process_noise = 1.0e-02 * np.eye(3)
    sensor_noise = np.eye(2)

    # Initial control
    u0 = ControlVariables(mean=np.array([0., 10, 28, 8./3., 2]))

    # Define UKF
    initial_state = KalmanHiddenState(mean=state0 + rng.multivariate_normal(mean=np.zeros(3), cov=cov_state),
                                      covariance=cov_state)
    ukf_chaos = UKF(model=lorenz_model,
                    initial_state=initial_state,
                    initial_control=u0,
                    covariance_process_noise=process_noise,
                    covariance_sensor_noise=sensor_noise)

    # Initialize dictionaries to store values
    real_values = {'state': [np.array([state0])],
                   'measurements': []}
    noisy_values = {'state': [np.array([state0 + rng.multivariate_normal(mean=np.zeros(3), cov=process_noise)])],
                    'measurements': []}
    ukf_values = {'state': [initial_state.model_copy(deep=True)],
                  'measurements': []}

    # Timestamps
    timestamps = [0.0]

    # Assign previous control
    previous_control = u0

    for _ in range(10000):
        # Get a new time
        time = timestamps[-1] + 0.01 * (1. + rng.random())
        timestamps += [time]
        # Choose random new controls
        sigma = 10
        beta = 8./3.
        rho = rng.normal(loc=28, scale=4)
        n = rng.integers(2, 5)
        u = ControlVariables(mean=np.array([time, sigma, rho, beta, n]))

        # Compute new true hidden state
        prev_hidden = noisy_values['state'][-1]
        new_state = lorenz_model.update_hidden_states(hidden_states=prev_hidden,
                                                      previous_controls=previous_control,
                                                      new_controls=u)
        real_values['state'] += [new_state.copy()]
        new_state += rng.multivariate_normal(mean=np.zeros(3), cov=process_noise)
        noisy_values['state'] += [new_state.copy()]

        # Get new measurement
        m = lorenz_model.predict_measurement(hidden_states=new_state, controls=u)
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
    m0_noise = np.array([y.flatten()[0]for y in noisy_values['measurements']])
    m1_noise = np.array([y.flatten()[1]for y in noisy_values['measurements']])

    # Check against UKF with error
    m0_1sig = np.isclose(m0_noise, ukf_m0, atol=1 * ukf_m0_std)
    m0_2sig = np.isclose(m0_noise, ukf_m0, atol=2 * ukf_m0_std)
    m0_3sig = np.isclose(m0_noise, ukf_m0, atol=3 * ukf_m0_std)
    m1_1sig = np.isclose(m1_noise, ukf_m1, atol=1 * ukf_m1_std)
    m1_2sig = np.isclose(m1_noise, ukf_m1, atol=2 * ukf_m1_std)
    m1_3sig = np.isclose(m1_noise, ukf_m1, atol=3 * ukf_m1_std)

    # Only consider the last few points, when the UKF should have converged
    last_points = 500
    m0_1sig_frac = np.sum(m0_1sig[-last_points:])/last_points
    m0_2sig_frac = np.sum(m0_2sig[-last_points:])/last_points
    m0_3sig_frac = np.sum(m0_3sig[-last_points:])/last_points
    m1_1sig_frac = np.sum(m1_1sig[-last_points:])/last_points
    m1_2sig_frac = np.sum(m1_2sig[-last_points:])/last_points
    m1_3sig_frac = np.sum(m1_3sig[-last_points:])/last_points

    # Check stats, but be slightly more lenient than a true Gaussian
    assert m0_1sig_frac >= 0.625, 'Fraction within 1 standard deviation is %2.1f %% < 62.5%%!!' % (100 * m0_1sig_frac)
    assert m0_2sig_frac >= 0.925, 'Fraction within 2 standard deviations is %2.1f %% < 92.5%%!!' % (100 * m0_2sig_frac)
    assert m0_3sig_frac >= 0.975, 'Fraction within 3 standard deviations is %2.1f %% < 97.5%%!!' % (100 * m0_3sig_frac)
    assert m1_1sig_frac >= 0.625, 'Fraction within 1 standard deviation is %2.1f %% < 62.5%%!!' % (100 * m1_1sig_frac)
    assert m1_2sig_frac >= 0.925, 'Fraction within 2 standard deviations is %2.1f < 92.5%%!!' % (100 * m1_2sig_frac)
    assert m1_3sig_frac >= 0.975, 'Fraction within 3 standard deviations is %2.1f %% < 97.5%%!!' % (100 * m1_3sig_frac)

    # Getting noisy hidden states
    x_noise = np.array([hid.flatten()[0]for hid in noisy_values['state']])
    y_noise = np.array([hid.flatten()[1]for hid in noisy_values['state']])
    z_noise = np.array([hid.flatten()[2]for hid in noisy_values['state']])

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
    x_1sig_frac = np.sum(x_1sig[-last_points:])/last_points
    x_2sig_frac = np.sum(x_2sig[-last_points:])/last_points
    x_3sig_frac = np.sum(x_3sig[-last_points:])/last_points
    y_1sig_frac = np.sum(y_1sig[-last_points:])/last_points
    y_2sig_frac = np.sum(y_2sig[-last_points:])/last_points
    y_3sig_frac = np.sum(y_3sig[-last_points:])/last_points
    z_1sig_frac = np.sum(z_1sig[-last_points:])/last_points
    z_2sig_frac = np.sum(z_2sig[-last_points:])/last_points
    z_3sig_frac = np.sum(z_3sig[-last_points:])/last_points

    # Check stats, but give a wider margin than a true Gaussian, as this is a very chaotic system and we are looking at
    # the hidden states, instead of number the UKF can measure itself against.
    assert x_1sig_frac > 0.60,  'Fraction within 1 standard deviation is %2.1f %% < 60%%!!' % (100 * x_1sig_frac)
    assert x_2sig_frac > 0.90,  'Fraction within 1 standard deviation is %2.1f %% < 90%%!!' % (100 * x_2sig_frac)
    assert x_3sig_frac > 0.95,  'Fraction within 1 standard deviation is %2.1f %% < 95%%!!' % (100 * x_3sig_frac)
    assert y_1sig_frac > 0.60,  'Fraction within 1 standard deviation is %2.1f %% < 60%%!!' % (100 * y_1sig_frac)
    assert y_2sig_frac > 0.90,  'Fraction within 1 standard deviation is %2.1f %% < 90%%!!' % (100 * y_2sig_frac)
    assert y_3sig_frac > 0.95,  'Fraction within 1 standard deviation is %2.1f %% < 95%%!!' % (100 * y_3sig_frac)
    assert z_1sig_frac > 0.60,  'Fraction within 1 standard deviation is %2.1f %% < 60%%!!' % (100 * z_1sig_frac)
    assert z_2sig_frac > 0.90,  'Fraction within 1 standard deviation is %2.1f %% < 90%%!!' % (100 * z_2sig_frac)
    assert z_3sig_frac > 0.95,  'Fraction within 1 standard deviation is %2.1f %% < 95%%!!' % (100 * z_3sig_frac)