from pytest import mark, raises
import numpy as np

from moirae.models.ecm import ECMInput
from moirae.simulator import Simulator


@mark.parametrize('batched', [True, False])
def test_dataframe(simple_rint, batched):
    """Make sure the simulator works with batched data"""
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint

    if batched:
        rint_asoh.r0.base_values = np.array([[0.05], [0.04]])
        rint_transient.from_numpy(np.repeat(rint_transient.to_numpy(), axis=0, repeats=2))
        assert rint_asoh.r0.base_values.shape == (2, 1)

    simulator = Simulator(
        model=ecm,
        asoh=rint_asoh,
        transient_state=rint_transient,
        initial_input=rint_inputs,
        keep_history=True
    )

    for time in np.arange(10.)[1:]:
        simulator.step(ECMInput(time=time, current=1.))

    df = simulator.to_dataframe()
    assert len(df) == 10 * rint_asoh.batch_size
    assert np.all(df.columns[:3] == ['batch', 'time', 'current'])

    assert np.allclose(df['time'].iloc[:rint_asoh.batch_size * 2:rint_asoh.batch_size], [0., 1.])

    # Test converting to batdata
    output = simulator.to_batdata(extra_columns=True)
    assert len(output) == rint_asoh.batch_size
    for batch in output:
        assert len(batch.validate()) == 0

    # Ensure that the sign convention is the HDF5 is opposite of the dataframe in the moirae convention
    assert len(output[0].raw_data) == len(df) // rint_asoh.batch_size
    assert np.allclose(output[0].raw_data['current'], -df.query('batch == 0')['current'])
    assert np.allclose(output[0].raw_data['test_time'], df.query('batch == 0')['time'])


def test_dataframe_failure(simple_rint):
    rint_asoh, rint_transient, rint_inputs, ecm = simple_rint

    simulator = Simulator(
        model=ecm,
        asoh=rint_asoh,
        transient_state=rint_transient,
        initial_input=rint_inputs,
        keep_history=False
    )

    with raises(ValueError, match='was not stored'):
        simulator.to_dataframe()
