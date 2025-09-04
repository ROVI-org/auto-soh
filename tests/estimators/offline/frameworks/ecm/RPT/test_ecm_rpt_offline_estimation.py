"""
Tests framework for offline estimation of ECM aSOH from RPT data
"""
from pytest import mark
import numpy as np

from moirae.interface import run_model
from moirae.models.ecm import EquivalentCircuitModel as ECM

from moirae.estimators.offline.frameworks.ecm.RPT import ECMOfflineEstimatorFromRPT
from moirae.estimators.offline.DataCheckers.RPT.hppc import FullHPPCCheckerPreinitParams
from moirae.estimators.offline.assemblers.ecm import ResistanceAssembler, OCVAssembler, HysteresisAssembler
from moirae.estimators.offline.assemblers.utils import SOCRegressor


@mark.slow
def test_offline_estimation_ecmasoh_rpt(realistic_LFP_aSOH, realistic_rpt_data):
    # Get the relevant parts of the data
    raw_rpt = realistic_rpt_data.tables['raw_data']
    cap_check = raw_rpt[raw_rpt['protocol'] == 'Capacity Check']

    # Prepare parameters for HPPC data checker
    hppc_params = FullHPPCCheckerPreinitParams(
        min_delta_soc=0.99,
        min_pulses=10,
        ensure_bidirectional=True,
        min_number_of_rests=10,
        # min_rest_duration=900.,
        # min_rest_prev_dur=300.
    )

    # Prepare assemblers
    ocv_assember = OCVAssembler(soc_points=np.array([0., 0.05, 0.1, 0.2, 0.75, 0.975, 1.0]))
    r0_assembler = ResistanceAssembler(regressor=SOCRegressor(style='interpolate', parameters={'k': 1}))
    # h0_assembler = HysteresisAssembler(regressor=SOCRegressor(style='interpolate', parameters={'k': 0}),
    #                                     soc_points=np.array([0.0, 0.1, 0.3, 0.4, 0.6, 0.7, 1.0]))
    h0_assembler = HysteresisAssembler(regressor=SOCRegressor(style='lsq', parameters={'k': 1}),
                                    soc_points=[-0.01, 0.0, 0.1, 0.3, 0.4, 0.6, 0.7, 1.0, 1.01, 1.05])

    # Initialize offline estimator
    offline_estimator = ECMOfflineEstimatorFromRPT(hppc_checker_params=hppc_params)

    transient, asoh, result = offline_estimator.estimate(
        data=raw_rpt,
        capacity_check_cycle_number=1,
        hppc_test_cycle_number=2,
        start_soc=raw_rpt['SOC'].iloc[0],
        number_rc_pairs=1,
        asoh_assemblers={'r0': r0_assembler, 'ocv': ocv_assember, 'hyst': h0_assembler},
        params_to_refine=['h0.base_values'],
        minimizer_kwargs={'method':'Nelder-Mead',
                          'tol': 1.0e-02,
                          'options': dict(maxiter=40),
                          }
        )

    # Now, check that the minimization was successful (it takes ~7 minutes)
    assert result.success, "Miminization not sucessful"

    # Compare predictions
    soc_test = np.linspace(0, 1, 101)
    assert np.allclose(asoh.q_t.amp_hour, realistic_LFP_aSOH.q_t.amp_hour, rtol=0.01), \
        'Maximum capacity discrepancy > 1%!'
    assert np.allclose(asoh.ce, realistic_LFP_aSOH.ce, atol=1.0e-06), 'Mismatch in CE!'
    assert np.allclose(asoh.ocv(soc_test), realistic_LFP_aSOH.ocv(soc_test), rtol=0.05), 'Mismatch in OCV!'
    assert np.allclose(asoh.r0.get_value(soc_test), realistic_LFP_aSOH.r0.get_value(soc_test), rtol=0.01), \
        'Mismatch in R0!'
    # Hysteresis works poorly at SOC extremes, and we should give it a few mV of tolerance as well
    soc_h0 = np.linspace(0.1, 1., 100)
    assert np.allclose(asoh.h0.get_value(soc_h0), realistic_LFP_aSOH.h0.get_value(soc_h0), rtol=0.05, atol=5.0e-03), \
        'Mismatch in H0!'
    for rc_est, rc_gt in zip(asoh.rc_elements, realistic_LFP_aSOH.rc_elements):
        assert np.allclose(rc_est.r.get_value(soc_test), rc_gt.r.get_value(soc_test), rtol=0.05), \
            "Mismatch in R_RC component!"
        assert np.allclose(rc_est.c.get_value(soc_test), rc_gt.c.get_value(soc_test), rtol=0.05), \
            "Mismatch in C_RC component!"

    # Now, let's simulate the RPT with this aSOH and compare
    simulation_results = run_model(model=ECM(),
                                   dataset=realistic_rpt_data,
                                   asoh=asoh,
                                   state_0=transient)
    v_errs = raw_rpt['voltage'] - simulation_results['terminal_voltage']
    mae = np.mean(abs(v_errs))
    rmse = np.sqrt(np.mean(np.pow(v_errs, 2)))
    assert mae < 0.01, f'Voltage MAE = {1000 * mae:.1f} mV > 10 mV'
    assert rmse < 0.015, f'Voltage MAE = {1000 * rmse:.1f} mV > 15 mV'
