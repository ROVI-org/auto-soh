from collections import defaultdict
from typing import Tuple

from pytest import fixture
from battdat.data import BatteryDataset, CellDataset
import pandas as pd
import numpy as np

from battdat.schemas import BatteryMetadata, BatteryDescription
from moirae.models.ecm import ECMASOH, ECMTransientVector, EquivalentCircuitModel, ECMInput
from moirae.simulator import Simulator


@fixture()
def simple_rint() -> Tuple[ECMASOH, ECMTransientVector, ECMInput, EquivalentCircuitModel]:
    """Get the parameters which define an uncharged R_int model"""

    return (ECMASOH.provide_template(has_C0=False, num_RC=0, H0=0.0),
            ECMTransientVector.provide_template(has_C0=False, num_RC=0),
            ECMInput(time=0., current=0.),
            EquivalentCircuitModel())


def make_dataset(simple_rint):
    rint_asoh, x, y, ecm_model = simple_rint

    # Run for a few cycles of (2C charge, rest, 1C discharge, rest)
    timestep = 5.
    num_cycles = 2
    charge_time = 1800
    discharge_time = charge_time * 2
    charge_current = rint_asoh.q_t.value.item() / charge_time
    discharge_current = -rint_asoh.q_t.value.item() / discharge_time
    rest_time = 20.

    # Run the model
    simulator = Simulator(
        EquivalentCircuitModel(),
        rint_asoh,
        initial_input=ECMInput(time=0., current=0.),
        transient_state=ECMTransientVector(soc=0.),
        keep_history=False
    )
    output = defaultdict(list)
    start_time = 0.

    def _update_outputs(time, current, transient, outputs, cycle_number):
        output['test_time'].append(time)
        output['current'].append(current)
        output['voltage'].append(outputs.terminal_voltage.item())
        output['cycle_number'].append(cycle_number)

    for cycle in range(num_cycles):
        for time in np.linspace(start_time, start_time + charge_time, int(charge_time / timestep) + 1):
            transient, outputs = simulator.step(ECMInput(time=time, current=charge_current))
            _update_outputs(time, charge_current, transient, outputs, cycle)
        start_time += charge_time

        for time in np.linspace(start_time, start_time + rest_time, int(rest_time / timestep) + 1):
            transient, outputs = simulator.step(ECMInput(time=time, current=0))
            _update_outputs(time, 0., transient, outputs, cycle)
        start_time += rest_time

        for time in np.linspace(start_time, start_time + discharge_time, int(discharge_time / timestep) + 1):
            transient, outputs = simulator.step(ECMInput(time=time, current=discharge_current))
            _update_outputs(time, discharge_current, transient, outputs, cycle)
        start_time += discharge_time

        for time in np.linspace(start_time, start_time + rest_time, int(rest_time / timestep) + 1):
            transient, outputs = simulator.step(ECMInput(time=time, current=0))
            _update_outputs(time, 0., transient, outputs, cycle)
        start_time += rest_time

    raw_data = pd.DataFrame(dict(output))

    # Make metadata with a cell capacity
    metadata = BatteryMetadata(
        battery=BatteryDescription(nominal_capacity=rint_asoh.q_t.amp_hour)
    )

    return CellDataset(raw_data=raw_data, metadata=metadata)


dataset = None


@fixture()
def timeseries_dataset(simple_rint) -> BatteryDataset:
    global dataset
    if dataset is None:
        dataset = make_dataset(simple_rint)
    return dataset


def test_timeseries(timeseries_dataset):
    timeseries_dataset.validate()
