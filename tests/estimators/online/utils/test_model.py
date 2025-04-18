import numpy as np

from moirae.estimators.online.utils.model import CellModelWrapper, DegradationModelWrapper


def test_dual_wrapper(simple_rint):
    # Get simple rint objects
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint

    # Cell wrapper
    cell_wrap = CellModelWrapper(cell_model=ecm, asoh=rint_asoh, transients=rint_transient, inputs=rint_inputs)
    assert cell_wrap.num_hidden_dimensions == 2
    assert cell_wrap.num_output_dimensions == 1

    # Now, make R0 updatable and with multiple values and check
    rint_asoh.r0.base_values = np.linspace(0.1, 1, 10).reshape((1, -1))
    rint_asoh.mark_updatable(name='r0.base_values')

    # Degradation wrapper
    deg_wrap = DegradationModelWrapper(cell_model=ecm, asoh=rint_asoh, transients=rint_transient, inputs=rint_inputs)
    assert deg_wrap.num_hidden_dimensions == 10
    assert deg_wrap.num_output_dimensions == 1
    deg_wrap._update_hidden_asoh(hidden_states=np.arange(1, 11))
    assert np.allclose(deg_wrap.asoh.get_parameters(), np.arange(1, 11).reshape((1, -1)))
