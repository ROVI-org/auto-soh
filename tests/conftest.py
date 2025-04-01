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


@fixture()
def ecm_rc() -> Tuple[ECMASOH, ECMTransientVector, ECMInput, EquivalentCircuitModel]:
    """Get the parameters which define an uncharged, single rc-element ecm model"""

    return (ECMASOH.provide_template(has_C0=False, num_RC=1, H0=0.0),
            ECMTransientVector.provide_template(has_C0=False, num_RC=1),
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
        battery=BatteryDescription(nominal_capacity=rint_asoh.q_t.amp_hour.item())
    )

    return CellDataset(raw_data=raw_data, metadata=metadata)


@fixture()
def timeseries_dataset(simple_rint) -> BatteryDataset:
    return make_dataset(simple_rint)


def test_timeseries(timeseries_dataset):
    timeseries_dataset.validate()


def make_dataset_hppc(model_and_params, ts=10):

    asoh, x, y, ecm_model = model_and_params

    simulator = Simulator(
        cell_model=EquivalentCircuitModel(), asoh=asoh,
        initial_input=ECMInput(),
        transient_state=ECMTransientVector.from_asoh(asoh),
        keep_history=True)

    # define key parameters of HPPC cycle
    tch = 36000  # charge time for 100pct DOD charge (s)
    Ich = asoh.q_t.value.item() / tch  # charge current
    tpulse = 10  # pulse time (s)
    Ipulse = 5  # pulse current (A)
    trestmp = 40  # mid-pulse rest (s)
    trestl = 3600  # long rest (s)
    Idi = Ich  # discharge current (A)
    tdi = 3600  # discharge time (s)

    # generate current time profiles for simulation input
    It_profile = {
        'cc_ch_100pctDoD': [tch, Ich],
        }
    for soc in np.arange(10, 101, 10)[::-1]:
        It_profile[f'prepulserest_{soc}pctSOC'] = [trestl, 0]
        It_profile[f'pulse_di_{soc}pctSOC'] = [tpulse, Ipulse]
        It_profile[f'midpulserest_{soc}pctSOC'] = [trestmp, 0]
        It_profile[f'pulse_ch_{soc}pctSOC'] = [tpulse, -Ipulse]
        It_profile[f'cc_di_{soc}pctSOC'] = [tdi, -Idi]
    It_profile['prepulserest_0pctSOC'] = [trestl, 0]
    It_profile['pulse_ch_0pctSOC'] = [tpulse, Ipulse]
    It_profile['midpulserest_0pctSOC'] = [trestmp, 0]
    It_profile['pulse_di_0pctSOC'] = [tpulse, -Ipulse]

    currents = []
    states = []
    tot_time = 0
    step_indices = []

    for ii, (key, value) in enumerate(It_profile.items()):
        t, curr = value

        n_ts = np.int32(np.floor(t/ts))  # number of time steps in t

        currents += [curr] * n_ts
        step_indices += [ii] * n_ts

        if curr > 0.0001:
            state = 'charging'
        elif curr < -0.0001:
            state = 'discharging'
        else:
            state = 'resting'

        states += [state] * n_ts
        tot_time += t

    timestamps = np.arange(1, tot_time+1, ts)
    timestamps = timestamps.tolist()

    # Prepare list of inputs
    ecm_inputs = [ECMInput(time=time, current=current)
                  for (time, current) in zip(timestamps, currents)]

    # Store results
    measurements = simulator.evolve(ecm_inputs)
    voltage = [measure.terminal_voltage.item() for measure in measurements]

    raw_data = pd.DataFrame({
        'test_time': timestamps,
        'current': currents,
        'voltage': voltage,
        'cycle_number': np.ones((len(voltage),)),
        'step_index': step_indices,
        'state': states
        }
    )

    # Make metadata with a cell capacity
    metadata = BatteryMetadata(
        battery=BatteryDescription(nominal_capacity=asoh.q_t.amp_hour.item())
    )

    # CellDataset(
    #     raw_data=raw_data, metadata=metadata).to_hdf(
    #         '../../docs/extractors/files/hppc_1rc.h5', complevel=9)

    return CellDataset(raw_data=raw_data, metadata=metadata)


@fixture()
def timeseries_dataset_hppc(simple_rint) -> BatteryDataset:
    return make_dataset_hppc(simple_rint, ts=10)


def test_timeseries_hppc(timeseries_dataset_hppc):
    timeseries_dataset_hppc.validate()


@fixture()
def timeseries_dataset_hppc_rc(ecm_rc) -> BatteryDataset:
    return make_dataset_hppc(ecm_rc, ts=1)


def test_timeseries_dataset_hppc_rc(timeseries_dataset_hppc_rc):
    timeseries_dataset_hppc_rc.validate()
