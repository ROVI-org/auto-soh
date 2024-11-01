import numpy as np

from moirae.models.ecm.components import Resistance


def test_r_batch():
    """Test that an SOC- and T-dependent health parameter works with batching"""

    # A single resistance value
    r_model = Resistance(base_values=0.05)
    r = r_model.get_value(0.)
    assert r.shape == (1, 1, 1)
    assert np.allclose(r, 0.05)

    r = r_model.get_value(np.array([0., 0.05]))
    assert r.shape == (1, 1, 2)
    assert np.allclose(r, 0.05)

    # Multiple resistance values
    r_model = Resistance(base_values=[0.05, 0.04])
    r = r_model.get_value(0.)
    assert r.shape == (1, 1, 1)
    assert np.allclose(r, 0.05)

    r = r_model.get_value(np.array([0., 0.5]))
    assert r.shape == (1, 1, 2)
    assert np.allclose(r, [[[0.05, 0.045]]])

    # Multiple resistance values, batched
    r_model = Resistance(base_values=[[0.05, 0.04], [0.06, 0.05]])
    assert r_model.batch_size == 2
    r = r_model.get_value(0.)
    assert r.shape == (2, 1, 1)
    assert np.allclose(r, [[[0.05]], [[0.06]]])

    r = r_model.get_value([0., 0.5])
    assert r.shape == (2, 1, 2)
    assert np.allclose(r, [[[0.05, 0.045]], [[0.06, 0.055]]])

    r = r_model.get_value([[0.], [0.5]])  # (n_batches, 1) format used elsewhere
    assert r.shape == (2, 2, 1)
    assert np.allclose(r, [[[0.05], [0.045]], [[0.06], [0.055]]])

    # SOC provided as "batched" with batch size of 1 (that is, shape is (1,1))
    r = r_model.get_value([[0.]])
    assert r.shape == (2, 1, 1)
    assert np.allclose(r, np.array([[[0.05]], [[0.06]]]))
