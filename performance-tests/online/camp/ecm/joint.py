"""Example estimator from:
https://github.com/ROVI-org/model-to-prognosis/blob/parallel-estimation/2_prepare-for-asoh-estimation.ipynb"""
from pathlib import Path

import numpy as np
from moirae.estimators.online.joint import JointEstimator
from moirae.models.ecm import EquivalentCircuitModel, ECMInput, ECMASOH, ECMTransientVector

# Load the initial ASOH
initial_asoh = ECMASOH.model_validate_json(Path('initial-asoh.json').read_text())

# Adjust reference OCV to include points outside the initial domain
initial_asoh.ocv(0.5)  # Initializes the interpolation points
soc_pinpoints = [-0.1] + initial_asoh.ocv.ocv_ref.soc_pinpoints.flatten().tolist() + [1.1]
base_vals = [0.] + initial_asoh.ocv.ocv_ref.base_values.flatten().tolist() + [6.5]
initial_asoh.ocv.ocv_ref.base_values = np.array([base_vals])
initial_asoh.ocv.ocv_ref.soc_pinpoints = np.array(soc_pinpoints)
initial_asoh.ocv.ocv_ref.interpolation_style = 'linear'

# Make it so the resistance and capacity will be estimated
initial_asoh.mark_updatable('r0.base_values')
initial_asoh.mark_updatable('q_t.base_values')

# Uncertainties for the parameters
# For A-SOH, assume 2*standard_dev is 0.5% of the value of the parameter
asoh_covariance = [(2.5e-03 * initial_asoh.q_t.base_values.item()) ** 2]  # +/- std_dev^2 Qt
asoh_covariance += ((2.5e-03 * initial_asoh.r0.base_values.flatten()) ** 2).tolist()  # +/- std_dev^2 of R0
asoh_covariance = np.diag(asoh_covariance)
# For the transients, assume SOC is a uniform random variable in [0,1], and hysteresis has 2*std_dev of 1 mV
init_transients = ECMTransientVector.from_asoh(initial_asoh)
init_transients.soc = np.atleast_2d(1.)
tran_covariance = np.diag([1 / 12, 2.5e-07])

# Make the noise terms
#  Logic from: https://github.com/ROVI-org/auto-soh/blob/main/notebooks/demonstrate_joint_ukf.ipynb
voltage_err = 1.0e-03  # mV voltage error
noise_sensor = ((voltage_err / 2) ** 2) * np.eye(1)
noise_asoh = 1.0e-10 * np.eye(asoh_covariance.shape[0])
noise_tran = 1.0e-08 * np.eye(2)

estimator = JointEstimator.initialize_unscented_kalman_filter(
    cell_model=EquivalentCircuitModel(),
    initial_asoh=initial_asoh.model_copy(deep=True),
    initial_inputs=ECMInput(
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
