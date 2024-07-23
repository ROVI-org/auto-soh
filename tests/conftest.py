from collections import defaultdict

import numpy as np
import pandas as pd
from pytest import fixture
from batdata.data import BatteryDataset

from asoh.models.ecm import EquivalentCircuitModel, ECMASOH, ECMInput
from asoh.models.ecm.simulator import ECMSimulator


@fixture()
def rint_model() -> ECMASOH:
    return ECMASOH.provide_template(has_C0=False, num_RC=0)


@fixture()
def timeseries_dataset(rint_model) -> BatteryDataset:
    # Run for a few cycles of (2C charge, rest, 1C discharge, rest)
    timestep = 1.
    num_cycles = 5
    charge_time = 1800
    discharge_time = charge_time * 2
    charge_current = rint_model.q_t.value / discharge_time
    discharge_current = -rint_model.q_t.value / discharge_time
    rest_time = 20.

    # Run the model
    ecm = EquivalentCircuitModel
    simulator = ECMSimulator(rint_model, keep_history=False)
    output = defaultdict(list)
    start_time = 0.

    def _update_outputs(time, current, transient, outputs):
        output['test_time'].append(time)
        output['current'].append(current)
        output['voltage'].append(outputs.terminal_voltage)

    for cycle in range(num_cycles):
        for time in np.linspace(start_time, start_time + charge_time, int(charge_time / timestep) + 1):
            transient, outputs = simulator.step(ECMInput(time=time, current=charge_current))
            _update_outputs(time, charge_current, transient, outputs)
        start_time += charge_time

        for time in np.linspace(start_time, start_time + rest_time, int(rest_time / timestep) + 1):
            transient, outputs = simulator.step(ECMInput(time=time, current=0))
            _update_outputs(time, 0., transient, outputs)
        start_time += rest_time

        for time in np.linspace(start_time, start_time + discharge_time, int(discharge_time / timestep) + 1):
            transient, outputs = simulator.step(ECMInput(time=time, current=discharge_current))
            _update_outputs(time, discharge_current, transient, outputs)
        start_time += discharge_time

        for time in np.linspace(start_time, start_time + rest_time, int(rest_time / timestep) + 1):
            transient, outputs = simulator.step(ECMInput(time=time, current=0))
            _update_outputs(time, 0., transient, outputs)
        start_time += rest_time

    raw_data = pd.DataFrame(dict(output))
    return BatteryDataset(raw_data=raw_data)


def test_timeseries(timeseries_dataset):
    timeseries_dataset.validate()
