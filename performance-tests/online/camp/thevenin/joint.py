"""Example estimator from:
https://github.com/ROVI-org/model-to-prognosis/blob/parallel-estimation/2_prepare-for-asoh-estimation.ipynb"""
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyfit
from moirae.estimators.online.joint import JointEstimator
from moirae.models.ecm import ECMASOH
from moirae.models.thevenin import TheveninASOH, TheveninModel
from moirae.models.thevenin.state import TheveninTransient
from moirae.models.thevenin.ins_outs import TheveninInput
from moirae.models.thevenin.components import SOCPolynomialVariable, SOCTempPolynomialVariable

# Load the initial ASOH used in the ECM model
ecm_asoh = ECMASOH.model_validate_json(Path('../ecm/initial-asoh.json').read_text())

# Translate it to Thevenin's formats
ecm_asoh.ocv(0.5)  # Initializes the interpolation points
ocv_poly = polyfit(
    x=ecm_asoh.ocv.ocv_ref.soc_pinpoints,
    y=ecm_asoh.ocv.ocv_ref.base_values[0, :],
    deg=4
)

ecm_asoh.r0.get_value(0.5)
r0_poly = polyfit(
    x=ecm_asoh.r0.soc_pinpoints,
    y=ecm_asoh.r0.base_values[0, :],
    deg=2
)

initial_asoh = TheveninASOH(
    capacity=ecm_asoh.q_t.base_values.item(),
    ocv=SOCPolynomialVariable(coeffs=ocv_poly),
    r=(
        SOCTempPolynomialVariable(soc_coeffs=r0_poly, t_coeffs=0),
    ),
)


# Make it so the resistance and capacity will be estimated
initial_asoh.mark_updatable('r.0.soc_coeffs')
initial_asoh.mark_updatable('capacity')

# Uncertainties for the parameters
# For A-SOH, assume 2*standard_dev is 0.5% of the value of the parameter
asoh_covariance = [(2.5e-03 * initial_asoh.capacity.item()) ** 2]  # +/- std_dev^2 Qt
asoh_covariance += ((2.5e-03 * initial_asoh.r[0].soc_coeffs[0, :]) ** 2).tolist()  # +/- std_dev^2 of R0
asoh_covariance = np.diag(asoh_covariance)

# For the transients, assume SOC is a uniform random variable in [0,1], and hysteresis has 2*std_dev of 1 mV
init_transients = TheveninTransient.from_asoh(initial_asoh)
init_transients.soc = np.atleast_2d(1.)
tran_covariance = np.diag([1 / 12, 1.])

# Make the noise terms
#  Logic from: https://github.com/ROVI-org/auto-soh/blob/main/notebooks/demonstrate_joint_ukf.ipynb
voltage_err = 1.0e-03  # mV voltage error
noise_sensor = ((voltage_err / 2) ** 2) * np.eye(1)
noise_asoh = 1.0e-10 * np.eye(asoh_covariance.shape[0])
noise_tran = 1.0e-08 * np.eye(2)

estimator = JointEstimator.initialize_unscented_kalman_filter(
    cell_model=TheveninModel(),
    initial_asoh=initial_asoh.model_copy(deep=True),
    initial_inputs=TheveninInput(
        time=0,
        current=0,
    ),
    initial_transients=init_transients,
    covariance_asoh=asoh_covariance,
    covariance_transient=tran_covariance,
    transient_covariance_process_noise=noise_tran,
    asoh_covariance_process_noise=noise_asoh,
    covariance_sensor_noise=noise_sensor
)
