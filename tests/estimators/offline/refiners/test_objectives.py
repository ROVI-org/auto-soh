from scipy.stats import norm
import numpy as np

from pytest import mark
from moirae.estimators.offline.refiners.loss import MeanSquaredLoss, PriorLoss, AdditiveLoss
from moirae.models.ecm import ECMInput, ECMMeasurement


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
        input_class=ECMInput,
        output_class=ECMMeasurement
    )

    # Run the evaluation
    x0 = loss.get_x0()[None, :]
    y = loss(x0, observations=timeseries_dataset)
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
    )

    # Run the evaluation
    x0 = loss.get_x0()[None, :]
    y = loss(x0, observations=timeseries_dataset)
    assert y.shape == (1,)
    qt_logprob = qt_dist.logpdf(10.)
    hy_logprob = hy_dist.logpdf(0.)
    assert np.isclose(y, -hy_logprob if state_only else -(qt_logprob + hy_logprob))


def test_additive(simple_rint, timeseries_dataset):
    rint_asoh, rint_state, _, ecm_model = simple_rint

    # Define the two loss functions
    hy_dist = norm(loc=0, scale=0.1)
    prior_loss = PriorLoss(
        transient_priors={'hyst': hy_dist},
        asoh_priors={},
        cell_model=ecm_model,
        asoh=rint_asoh,
        transient_state=rint_state,
    )
    fit_loss = MeanSquaredLoss(
        cell_model=ecm_model,
        asoh=rint_asoh,
        transient_state=rint_state
    )
    loss = AdditiveLoss(
        losses=[(0.1, prior_loss), (1, fit_loss)]
    )

    # Run the evaluation
    x0 = loss.get_x0()[None, :]
    y0_fit = fit_loss(x0, observations=timeseries_dataset)
    y = loss(x0, observations=timeseries_dataset)
    assert y.shape == (1,)
    hy_logprob = -hy_dist.logpdf(0.)
    assert np.allclose(y, 0.1 * hy_logprob + y0_fit)
