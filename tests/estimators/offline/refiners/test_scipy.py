import numpy as np
from pytest import mark

from moirae.estimators.offline.refiners.loss import MeanSquaredLoss
from moirae.estimators.offline.refiners.scipy import ScipyMinimizer, ScipyDifferentialEvolution


@mark.parametrize('state_only', [True, False])
@mark.parametrize('minimizer', [(ScipyMinimizer, {}),
                                (ScipyDifferentialEvolution, {'maxiter': 1, 'rng': 1})])
def test_scipy(simple_rint, timeseries_dataset, state_only, minimizer):
    rint_asoh, rint_state, _, ecm_model = simple_rint

    # Truncate the battery dataset
    timeseries_dataset.tables['raw_data'] = timeseries_dataset.tables['raw_data'].head(32)

    # Alter the estimated state slightly
    state_0 = rint_state.make_copy(np.array([[0.05, 0.]]))
    bounds = [(0, 1), (-1e-3, 1e-3)]

    if not state_only:
        rint_asoh.mark_updatable('q_t.base_values')
        bounds.append((0, 100))
        assert rint_asoh.num_updatable == 1
    loss = MeanSquaredLoss(
        cell_model=ecm_model,
        asoh=rint_asoh,
        transient_state=state_0
    )

    # Run the optimizer
    cls, kwargs = minimizer
    scipy = cls(loss, **kwargs, bounds=bounds)
    state, asoh, result = scipy.refine(observations=timeseries_dataset)
    assert np.allclose(state.to_numpy(), rint_state.to_numpy(), atol=1e-1)
    if not state_only:
        # Not much data from which to judge capacity from our early cycle-measurement
        assert np.allclose(asoh.get_parameters(), rint_asoh.get_parameters(), atol=1e-1)
