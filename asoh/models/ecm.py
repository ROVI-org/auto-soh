from progpy.prognostics_model import PrognosticsModel


class SingleResistor(PrognosticsModel):
    """A battery system modeled by a single resistor and open-circuit voltage which depends only on state of charge.

    State Variables
    --------------

    soc: State of charge of the battery system. Units: A-hr
    r_int: Internal resistance of the battery. Units: Ohm

    Input Variables
    ---------------

    i: Current applied to the system. Units: A

    Output Variables
    ----------------

    v: Observed voltage. Units: V

    """

    states = [
        'soc', 'r_int', 'ocv_0', 'ocv_1'
    ]
    inputs = ['i']

    default_parameters = {
        'x0': {
            # State of charge parameters
            'soc': 0,  # Start uncharged

            # State of health parameters
            'r_int': 1,  # Resistance in Ohm,
            'ocv_0': 1,  # Parameters of the OCV model
            'ocv_1': 0  # Parameters of the OCV model
        }
    }

    @property
    def state_params(self) -> list[str]:
        """Parameters which define the state of charge of the system"""
        return ['soc']

    @property
    def soh_params(self):
        """Parameters which define the state of health of the system"""
        return [x for x in self.default_parameters['x0'] if x not in self.state_params]

    def dx(self, x, u):
        # Update the SOC based on current
        dx = {'soc': u['i'] / 3600}  # SOC increases with charge

        # Set dx of the SOH-related variables to zero
        #  Nothing changes for them
        for p in self.soh_params:
            dx[p] = 0

        return self.StateContainer(dx)

    def output(self, x):
        return self.OutputContainer({
            'v': x['ocv_0'] + x['ocv_1'] * x['soc']
        })
