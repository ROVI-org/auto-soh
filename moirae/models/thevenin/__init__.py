"""Interface to the `Thevenin package <https://github.com/ROVI-org/thevenin>`_"""
import numpy as np
from functools import partial
from typing import Iterator

from thevenin import Model, Experiment

from ..base import CellModel, InputQuantities, OutputQuantities
from .state import TheveninASOH, TheveninTransient
from .ins_outs import TheveninInput


class TheveninModel(CellModel):
    """
    Connection between Moirae and an ECM model implemented via `Thevenin <https://github.com/ROVI-org/thevenin>`_.

    Args:
        isothermal: Whether to treat the system as isothermal
    """

    def __init__(self, isothermal: bool = False):
        self.isothermal = isothermal
        # TODO (wardlt): Add a maximum timestep size? Victor was using that to avoid the issues with the integrator

    def _make_models(self, transient: TheveninTransient, asoh: TheveninASOH, inputs: TheveninInput) -> Iterator[Model]:
        """
        Generate a model for each member of the batch of experimental conditions

        Args:
            transient: Batch of transient states
            asoh: Batch of health variables

        Yields:
            A model for each member of the batch
        """

        for b in range(max(transient.batch_size, asoh.batch_size, inputs.batch_size)):
            # The value of each member of the transient or ASOH is a 2D array with the first dimension either 1
            #  or batch_size. The % signs below are a short syntax for either using the same value for all batches
            #  (anything mod 1 is 0) or the appropriate member of the batch
            params = {'num_RC_pairs': asoh.num_rc_elements, 'isothermal': self.isothermal}
            for scalar, value in [
                ('soc0', transient.soc), ('capacity', asoh.capacity), ('mass', asoh.mass), ('Cp', asoh.c_p),
                ('T_inf', inputs.t_inf), ('h_therm', asoh.h_thermal), ('A_therm', asoh.a_therm)
            ]:
                params[scalar] = value[b % value.shape[0], 0]

            # Add the SOC and series resistors as functions where we pin the batch ID to the appropriate value
            params['ocv'] = partial(asoh.ocv, batch_id=b)
            params['R0'] = partial(asoh.r[0], batch_id=b)

            # Append the RC elements
            for r in range(params['num_RC_pairs']):
                params[f'R{r + 1}'] = partial(asoh.r[r + 1], batch_id=b)
                params[f'C{r + 1}'] = partial(asoh.r[r], batch_id=b)

            # Make the model
            model = Model(params=params)

            # Update the state of the RC elements
            #  TODO (wardlt): Working with Corey on an official route for setting these SVs.
            #   This is his current recommendation
            if params['num_RC_pairs'] > 0:
                model._sv0[model._ptr['eta_j']] = transient.eta[b % transient.eta.shape[0]]

            yield model

    def update_transient_state(
            self,
            previous_inputs: TheveninInput,
            new_inputs: TheveninInput,
            transient_state: TheveninTransient,
            asoh: TheveninASOH
    ) -> TheveninTransient:
        # Initialize the array in which to store output values
        batch_size = max(transient_state.batch_size, asoh.batch_size, new_inputs.batch_size)
        output_array = np.zeros((batch_size, len(transient_state)))

        # Iterate over models representing each member of the batch
        for model_i, model in enumerate(self._make_models(transient_state, asoh, new_inputs)):
            # Propagate the system under a constant current load
            # TODO (wardlt): Make current time-dependent, which is possible by passing a function to add_step
            exp = Experiment()
            cur_time = new_inputs.time[model_i % new_inputs.time.shape[0], 0]
            pre_time = previous_inputs.time[model_i % previous_inputs.time.shape[0], 0]
            exp.add_step('current_A',
                         -new_inputs.current[model_i % new_inputs.current.shape[0]],  # Sign convention is opposite
                         (cur_time - pre_time, 2))
            sln = model.run(exp)

            # Fill in the state variables
            output_array[model_i, 0] = sln.vars['soc'][-1]
            output_array[model_i, 1] = sln.vars['temperature_K'][-1]
            for rc_i in range(asoh.num_rc_elements):
                output_array[model_i, 2 + rc_i] = sln.vars[f'eta{rc_i + 1}_V'][-1]

        return transient_state.make_copy(output_array)

    def calculate_terminal_voltage(
            self,
            new_inputs: InputQuantities,
            transient_state: TheveninTransient,
            asoh: TheveninASOH) -> OutputQuantities:
        # Thevenin stores overpotentials, so it is easy enough to compute terminal voltage directly from states and ASOH
        #  See last eq of https://rovi-org.github.io/thevenin/user_guide/model_description.html
        v = (
                asoh.ocv(transient_state.soc[:, 0])
                # Sign convention is opposite of thevenin
                + new_inputs.current[:, 0] * asoh.r[0](transient_state.soc[:, 0], transient_state.temp[:, 0])
                - transient_state.eta.sum(axis=1, keepdims=True)
        )
        return OutputQuantities(terminal_voltage=v)
