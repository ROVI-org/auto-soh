import numpy as np
from pytest import mark

from moirae.estimators.offline.objectives import MeanSquaredLoss
from moirae.estimators.offline.scipy import ScipyMinimizer


@mark.parametrize('state_only', [True, False])
def test_scipy(simple_rint, timeseries_dataset, state_only):
    rint_asoh, rint_state, _, ecm_model = simple_rint

    # Truncate the battery dataset
    timeseries_dataset.raw_data = timeseries_dataset.raw_data.head(32)

    if not state_only:
        rint_asoh.mark_updatable('q_t.base_values')
        assert rint_asoh.num_updatable == 1
    loss = MeanSquaredLoss(
        cell_model=ecm_model,
        asoh=rint_asoh,
        state=rint_state,
        observations=timeseries_dataset
    )

    # Alter the estimated state slightly
    state_0 = rint_state.make_copy(np.array([[0.05, 0.]]))

    # Run the optimizer
    scipy = ScipyMinimizer(loss)
    state, asoh, result = scipy.estimate(state_0=state_0)
    assert np.allclose(state.to_numpy(), rint_state.to_numpy(), atol=1e-3)
    if not state_only:
        # Not much data from which to judge capacity from our early cycle-measurement
        assert np.allclose(asoh.get_parameters(), rint_asoh.get_parameters(), atol=1e-1)
