import numpy as np

from pytest import mark
from moirae.estimators.offline.loss import MeanSquaredLoss


@mark.parametrize('state_only', [True, False])
def test_mse_loss(simple_rint, timeseries_dataset, state_only):
    rint_asoh, rint_state, _, ecm_model = simple_rint

    if not state_only:
        rint_asoh.mark_updatable('q_t.base_values')
        assert rint_asoh.num_updatable == 1
    loss = MeanSquaredLoss(
        cell_model=ecm_model,
        asoh=rint_asoh,
        transient_state=rint_state,
        observations=timeseries_dataset
    )

    # Get a starting guess
    if state_only:
        x0 = rint_state.to_numpy()
    else:
        x0 = np.concatenate([rint_state.to_numpy(), rint_asoh.q_t.base_values], axis=1)

    # Run the evaluation
    y = loss(x0)
    assert y.shape == (1,)
    assert np.isclose(y, 0.)
