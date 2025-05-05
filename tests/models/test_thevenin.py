"""Test the interface to the Thevenin package"""

from pytest import mark
import numpy as np

from moirae.estimators.offline.loss import MeanSquaredLoss
from moirae.estimators.online.joint import JointEstimator
from moirae.models.base import OutputQuantities
from moirae.models.thevenin import TheveninInput, TheveninTransient, TheveninModel
from moirae.models.components.soc import SOCPolynomialHealth
from moirae.models.components.soc_t import SOCTempPolynomialHealth
from moirae.models.thevenin.state import TheveninASOH

rint = TheveninASOH(
    capacity=1.,
    ocv=SOCPolynomialHealth(coeffs=[1.5, 1.]),
    r=(SOCTempPolynomialHealth(soc_coeffs=[0.01, 0.01], t_coeffs=[0, 0.001]),)
)

rc2 = TheveninASOH(
    capacity=1.,
    ocv=SOCPolynomialHealth(coeffs=[1.5, 1.]),
    r=(SOCTempPolynomialHealth(soc_coeffs=[0.01, 0.01], t_coeffs=[0, 0.001]),) * 3,
    c=(SOCTempPolynomialHealth(soc_coeffs=[10, 10], t_coeffs=[0]),) * 2
)


def test_rint():
    """Ensure the rint model's subcomponents work"""

    # Ensuring we can do SOC dependence
    assert rint.ocv.get_value(0.5, batch_id=0) == 2.

    # Ensuring we can do SOC and temperature dependence
    assert rint.r[0].get_value(0.5, 25, batch_id=0) == 0.015
    assert rint.r[0].get_value(0.5, 35, batch_id=0) == 0.025

    # Test a single step at constant current
    state = TheveninTransient(soc=0., cell_temperature=25.)
    pre_inputs = TheveninInput(current=1., time=0., t_inf=25.)
    new_inputs = TheveninInput(current=1., time=30., t_inf=25.)

    model = TheveninModel()
    new_state = model.update_transient_state(pre_inputs, new_inputs, state, rint)
    assert new_state.batch_size == 1
    assert np.allclose(new_state.soc.item(), 30 / 3600.)
    assert new_state.cell_temperature > state.cell_temperature  # Solving for the actual answer is annoying

    # Get the terminal voltage
    voltage = model.calculate_terminal_voltage(new_inputs, new_state, rint)
    assert voltage.batch_size == 1
    assert np.allclose(voltage.terminal_voltage, (1.5 + 30 / 3600) + rint.r[0].get_value(30 / 3600, 25))


@mark.parametrize('asoh', [rint, rc2])
def test_multiple_steps(asoh):
    """Make sure the code works with multiple steps per charge and discharge cycle"""

    # Test a single step at constant current
    state = TheveninTransient.from_asoh(asoh)
    pre_inputs = TheveninInput(current=1., time=0., t_inf=25.)
    new_inputs = TheveninInput(current=1., time=30., t_inf=25.)

    # Test a single step of 30s charging
    model = TheveninModel()
    state = model.update_transient_state(pre_inputs, new_inputs, state, asoh)
    assert len(state) == 3 + asoh.num_rc_elements
    assert not np.isclose(state.to_numpy()[:, :2], 0.).any()  # Including the SOC and RC elements

    # Ensure no errors if the time between timesteps is zero
    new_state = model.update_transient_state(new_inputs, new_inputs, state, asoh)
    assert np.allclose(new_state.to_numpy(), state.to_numpy())

    # Test charging until the full hour
    pre_inputs = new_inputs
    new_inputs = TheveninInput(current=1., time=3600., t_inf=25.)
    state = model.update_transient_state(pre_inputs, new_inputs, state, asoh)
    assert np.allclose(state.soc, 1.)
    assert np.less(state.eta, 0.).all()
    assert np.greater(state.cell_temperature, 25.).all()

    v = model.calculate_terminal_voltage(new_inputs, state, asoh)
    assert np.greater_equal(v.terminal_voltage, 2.5 + 1 * 0.02).all()

    # Rest for 15 minutes, so that all RC elements and temperature equilibrate
    pre_inputs = new_inputs
    new_inputs = TheveninInput(current=0., time=pre_inputs.time + 15 * 60., t_inf=25.)
    state = model.update_transient_state(pre_inputs, new_inputs, state, asoh)
    assert np.allclose(state.soc, 1.)
    assert np.allclose(state.eta, 0.)
    assert np.isclose(state.cell_temperature, 25.)

    v = model.calculate_terminal_voltage(new_inputs, state, asoh)
    assert np.isclose(v.terminal_voltage, 2.5).all()

    # Discharge for an hour to get back to SOC 0
    pre_inputs = new_inputs
    new_inputs = TheveninInput(current=-1, time=pre_inputs.time + 3600., t_inf=25.)
    state = model.update_transient_state(pre_inputs, new_inputs, state, asoh)
    assert np.allclose(state.soc, 0.)
    assert np.greater(state.eta, 0.).all()
    assert np.greater(state.cell_temperature, 25.).all()

    v = model.calculate_terminal_voltage(new_inputs, state, asoh)
    assert np.less_equal(v.terminal_voltage, 1.5 - 1 * 0.01).all()

    # Rest for 15 minutes, so that all RC elements and temperature equilibrate
    pre_inputs = new_inputs
    new_inputs = TheveninInput(current=0., time=pre_inputs.time + 15 * 60., t_inf=25.)
    state = model.update_transient_state(pre_inputs, new_inputs, state, asoh)
    assert np.allclose(state.soc, 0.)
    assert np.allclose(state.eta, 0.)
    assert np.isclose(state.cell_temperature, 25.)

    v = model.calculate_terminal_voltage(new_inputs, state, asoh)
    assert np.isclose(v.terminal_voltage, 1.5).all()


def test_batching():
    """Evaluate whether batches of ASOH coordinates are treated properly"""

    # Run an experiment where the serial resistances are increasing
    model = TheveninModel()
    asoh = rint.model_copy(deep=True)
    asoh.r[0].soc_coeffs = np.array([[0.01], [0.02], [0.03]])
    assert asoh.batch_size == 3

    # The temperature of the cell should be higher with higher resistance
    state = TheveninTransient.from_asoh(asoh)
    pre_inputs = TheveninInput(current=1., time=0., t_inf=25.)
    new_inputs = TheveninInput(current=1., time=30., t_inf=25.)

    new_state = model.update_transient_state(pre_inputs, new_inputs, state, asoh)
    assert new_state.batch_size == 3
    assert (np.diff(new_state.cell_temperature[:, 0]) > 0).all()  # Temperatures should be increasing
    assert np.std(new_state.soc) < 1e-6  # SOC should all be the same

    # Compute the voltage
    soc = 1. / 120
    v = 1.5 + soc + 1 * asoh.r[0].soc_coeffs[:, 0]
    pred_v = model.calculate_terminal_voltage(new_inputs, new_state, asoh)
    assert np.allclose(v, pred_v.terminal_voltage[:, 0], atol=1e-3)  # Differences are due to temp


def test_estimator():
    """Make sure everything functions with an estimator"""

    # Test a single step at constant current
    asoh = rint.model_copy(deep=True)
    state = TheveninTransient.from_asoh(asoh)
    pre_inputs = TheveninInput(current=1., time=0., t_inf=25.)
    new_inputs = TheveninInput(current=1., time=30., t_inf=25.)

    model = TheveninModel(isothermal=True)

    # Make a joint estimator with the R0 as an adjustable parameter so that we
    #  get batching on at least one variable
    asoh.mark_updatable('r.0.soc_coeffs')
    est = JointEstimator.initialize_unscented_kalman_filter(
        cell_model=model,
        initial_asoh=asoh,
        initial_transients=state,
        initial_inputs=pre_inputs,
        covariance_transient=np.diag([0.05, 0.1, 1e-6]),
        covariance_asoh=np.diag([1e-3] * 2),
        transient_covariance_process_noise=np.diag([0.01] * 3),
        asoh_covariance_process_noise=np.diag([1e-3] * 2),
        covariance_sensor_noise=np.diag([1e-3])
    )
    assert est.state.get_covariance().shape == (5, 5)

    # Write down the expected results after 30 seconds of a 1A charge
    soc = 1. / 120
    expected_state = [soc, 25., 0]
    expected_v = 1.5 + soc + 1 * (0.01 * (1 + soc))

    est_state, est_outputs = est.step(new_inputs, OutputQuantities(terminal_voltage=expected_v))
    assert np.isclose(est_outputs.get_mean(), expected_v)
    est_mean = est_state.get_mean()
    assert np.allclose(expected_state, est_mean[:3], rtol=1e-4)  # SOC, T, hyst
    assert np.allclose(asoh.r[0].soc_coeffs, est_mean[3:], atol=1e-6)  # Temp dependence of R

    # Perturb the r0 and see if it converges back to the correct value
    #  Run several cycles of 1 A charge/discharge
    asoh.r[0].soc_coeffs[0, 0] = 0.011
    state.soc = np.atleast_2d(soc)
    est = JointEstimator.initialize_unscented_kalman_filter(
        cell_model=model,
        initial_asoh=asoh,
        initial_transients=state,
        initial_inputs=pre_inputs,
        covariance_transient=np.diag([0.001, 0.1, 1e-6]),
        covariance_asoh=np.diag([1e-3] * 2),
        transient_covariance_process_noise=np.diag([0.01] * 3),
        asoh_covariance_process_noise=np.diag([1e-3] * 2),
        covariance_sensor_noise=np.diag([1e-3])
    )
    assert np.isclose(est.get_estimated_state()[1].r[0].soc_coeffs[0, 0], 0.011)

    def _soc_over_time(t: float):
        """Actual SOC as a function of time"""
        time_in_cycle = t % 7200
        return time_in_cycle / 3600 if time_in_cycle < 3600 else 1 - (time_in_cycle - 3600) / 3600

    def _iv_over_time(t: float):
        time_in_cycle = t % 7200
        current = 1 if time_in_cycle < 3600 else -1
        soc = _soc_over_time(t)
        return current, 1.5 + soc + current * 0.01 * (1 + soc)

    for time in np.linspace(0, 7200 * 2, 500):
        i, v = _iv_over_time(time)
        est.step(
            TheveninInput(current=i, time=time, t_inf=298),
            OutputQuantities(terminal_voltage=v)
        )
    est_tran, est_asoh = est.get_estimated_state()
    assert np.isclose(est_tran.soc, 0., atol=0.05)


def test_overpotentials():
    """Ensure solution of the RC overpotentials is correct"""

    rc = TheveninASOH(
        capacity=1.,
        ocv=SOCPolynomialHealth(coeffs=[1.5, 1.]),
        r=(
            SOCTempPolynomialHealth(soc_coeffs=0.010, t_coeffs=0),
            SOCTempPolynomialHealth(soc_coeffs=0.020, t_coeffs=0),
            SOCTempPolynomialHealth(soc_coeffs=0.030, t_coeffs=0),
        ),
        c=(
            SOCTempPolynomialHealth(soc_coeffs=[1000], t_coeffs=[0]),
            SOCTempPolynomialHealth(soc_coeffs=[2000], t_coeffs=[0])
        )
    )
    assert rc.c[0].get_value(0., 298) == 1000
    assert rc.c[1].get_value(0., 298) == 2000

    # Charge for 10s at 1A
    state = TheveninTransient.from_asoh(rc)
    pre_inputs = TheveninInput(current=1., time=0., t_inf=25.)
    new_inputs = TheveninInput(current=1., time=10., t_inf=25.)

    model = TheveninModel()
    new_state = model.update_transient_state(pre_inputs, new_inputs, state, rc)
    assert new_state.eta.shape == (1, 2)
    # The eta_i should be: I r_i (1 - exp(-t / r_i / c_i)
    assert np.allclose(new_state.eta, [[
        -0.02 * (1 - np.exp(-10 / 0.02 / 1000)),
        -0.03 * (1 - np.exp(-10 / 0.03 / 2000))
    ]], atol=1e-4)


def test_offline(timeseries_dataset):
    """Ensure thevenin works with the core offline estimator objectives"""
    loss = MeanSquaredLoss(
        cell_model=TheveninModel(isothermal=True),
        asoh=rint,
        transient_state=TheveninTransient.from_asoh(rint),
        observations=timeseries_dataset,
        input_class=TheveninInput,
        output_class=OutputQuantities
    )

    # Run the evaluation
    x0 = loss.get_x0()[None, :]
    y = loss(x0)
    assert y.shape == (1,)
