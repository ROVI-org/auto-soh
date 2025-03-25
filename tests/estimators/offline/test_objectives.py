from scipy.stats import norm
import numpy as np

from pytest import mark
from moirae.estimators.offline.loss import MeanSquaredLoss, PriorLoss


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


@mark.parametrize('state_only', [True, False])
def test_prior_loss(simple_rint, timeseries_dataset, state_only):
    rint_asoh, rint_state, _, ecm_model = simple_rint
    if not state_only:
        rint_asoh.mark_updatable('q_t.base_values')
        assert rint_asoh.num_updatable == 1

    # Assume the q_t with a normal distribution around 10 with a std of 1,
    #  and the hyst with a mean of 0 and std of 0.1
    qt_dist = norm(loc=10, scale=1)
    hy_dist = norm(loc=0, scale=0.1)

    loss = PriorLoss(
        transient_priors={'hyst': hy_dist},
        asoh_priors={} if state_only else {'q_t.base_values': qt_dist},
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
    qt_logprob = qt_dist.logpdf(10.)
    hy_logprob = hy_dist.logpdf(0.)
    assert np.isclose(y, -hy_logprob if state_only else -(qt_logprob + hy_logprob))
