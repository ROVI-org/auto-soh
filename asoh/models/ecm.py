from progpy.utils.containers import StateContainer

from asoh.models.base import HealthModel


class SingleResistor(HealthModel):
    """A battery system modeled by a single resistor and open-circuit voltage which depends only on state of charge.

    State Variables
    --------------

    soc: State of charge of the battery system. Units: A-hr
    r_int: Internal resistance of the battery. Units: Ohm
    ocv_0: Constant component of open circuit voltage. Units: V
    ocv_1: Component of open circuit voltage which changes linearly with state of charge. Units: V/A-hr

    Input Variables
    ---------------

    i: Current applied to the system. Units: A

    Output Variables
    ----------------

    v: Observed voltage. Units: V

    """

    inputs = ['i']

    _health_parameters = {
        'r_int': 1,  # Resistance in Ohm,
        'ocv_0': 1,  # Parameters of the OCV model
        'ocv_1': 0  # Parameters of the OCV model
    }

    _always_states = {
        'soc': 0,  # Start uncharged
    }

    def dx(self, x, u):
        # Get dx/dt of health parameters (0)
        dx = super().dx(x, u)

        # Update the SOC based on current
        dx['soc'] = u['i'] / 3600  # SOC increases with charge

        return self.StateContainer(dx)

    def output(self, x, u=None):
        return self.OutputContainer({
            'v': x['ocv_0'] + x['ocv_1'] * x['soc'] + x['r_int'] * u['i']
        })
