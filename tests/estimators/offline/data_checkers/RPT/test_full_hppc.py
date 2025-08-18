from battdat.schemas.column import ControlMethod

from moirae.estimators.offline.DataCheckers.RPT import FullHPPCDataChecker


def test_full_hppc_checker(realistic_rpt_data, realistic_LFP_aSOH) -> None:
    """
    Tests full HPPC checker
    """
    hppc_checker = FullHPPCDataChecker(capacity=realistic_LFP_aSOH.q_t,
                                       coulombic_efficiency=realistic_LFP_aSOH.ce.item(),
                                       min_delta_soc=0.99,
                                       min_pulses=20,
                                       ensure_bidirectional=True,
                                       min_number_of_rests=12,
                                       min_rest_duration=900,
                                       rest_current_threshold=1.0e-04)
    # Get relevant data
    raw_rpt = realistic_rpt_data.tables.get('raw_data')
    hppc_data = raw_rpt[raw_rpt['protocol'] == 'Full HPPC']

    # Make sure it works
    checked_data = hppc_checker.check(data=hppc_data)
    checked_raw = checked_data.tables.get('raw_data')
    pulses = checked_raw[checked_raw['method'] == ControlMethod.pulse]

    assert len(pulses.groupby('substep_index')) == 20, f'Expected 20 pulses, but only {len(pulses)} we found!'
