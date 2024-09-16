import numpy as np

from pytest import mark
from moirae.estimators.offline.objectives import MeanSquaredLoss


@mark.parametrize('state_only', [True, False])
def test_mse_loss(simple_rint, timeseries_dataset, state_only):
    rint_asoh, rint_state, _, ecm_model = simple_rint

    if not state_only:
        rint_asoh.mark_updatable('q_t.base_values')
        assert rint_asoh.num_updatable == 1
    loss = MeanSquaredLoss(
        cell_model=ecm_model,
        asoh=rint_asoh,
        state=rint_state,
        observations=timeseries_dataset
    )

    # Test getting a starting guess
    x0 = loss.get_x0()
    assert x0.ndim == 1
    if state_only:
        assert np.allclose(x0, rint_state.to_numpy())
    else:
        assert np.allclose(x0, np.concatenate([rint_state.to_numpy(), rint_asoh.q_t.base_values], axis=1))
