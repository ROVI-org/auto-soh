"""Interface to the `Thevenin package <https://github.com/ROVI-org/thevenin>`_"""
import numpy as np
from functools import partial
from typing import Iterator

from thevenin import Prediction, TransientState

from ..base import CellModel, InputQuantities, OutputQuantities
from moirae.models.thevenin.state import TheveninASOH, TheveninTransient
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
        self._predictor = None  # Cache the predictor used between steps

    def _make_models(self, transient: TheveninTransient, asoh: TheveninASOH, inputs: TheveninInput) \
            -> Iterator[tuple[Prediction, TransientState]]:
        """
        Update the predictor and state for each member of the batch of experimental conditions

        Args:
            transient: Batch of transient states
            asoh: Batch of health variables
            inputs: Inputs to the battery system

        Yields:
            A Prediction class and Thevenin-compatible state for each member of the batch
        """

        for batch_id in range(max(transient.batch_size, asoh.batch_size, inputs.batch_size)):
            # Convert the parameters from Moira
            params, state = self._moirae_to_thevenin(batch_id, asoh, inputs, transient)

            # Make or update the model
            if self._predictor is None or self._predictor.num_RC_pairs != params['num_RC_pairs']:
                self._predictor = Prediction(params=params)
            else:
                for key, val in params.items():
                    setattr(self._predictor, key, val)

            yield self._predictor, state

    def _moirae_to_thevenin(self,
                            batch_id: int,
                            asoh: TheveninASOH,
                            inputs: TheveninInput,
                            transient: TheveninTransient) -> tuple[dict[str, float | np.ndarray], TransientState]:
        """Assemble Thevenin-compatible parameters for a single batch member
        from their description in Moirae

        ArgS:
            batch_id: Batch index
            asoh: Object holding the battery health parameters
            inputs: Inputs applied to the battery system
            transient: Transient state

        Returns:
            - Dictionary of parameters to applied to Predictor
            - Transient state of the battery
        """
        # The value of each member of the transient or ASOH is a 2D array with the first dimension either 1
        #  or batch_size. The % signs below are a short syntax for either using the same value for all batches
        #  (anything mod 1 is 0) or the appropriate member of the batch
        params = {'num_RC_pairs': asoh.num_rc_elements, 'isothermal': self.isothermal}
        for scalar, value in [
            ('capacity', asoh.capacity), ('mass', asoh.mass), ('Cp', asoh.c_p),
            ('T_inf', inputs.t_inf), ('h_therm', asoh.h_thermal), ('A_therm', asoh.a_therm), ('ce', asoh.ce),
            ('gamma', asoh.gamma), ('soc0', transient.soc)
        ]:
            params[scalar] = value[batch_id % value.shape[0], 0]

        # Add the SOC and series resistors as functions where we pin the batch ID to the appropriate value
        params['ocv'] = partial(asoh.ocv.get_value, batch_id=batch_id)
        params['R0'] = partial(asoh.r[0].get_value, batch_id=batch_id)
        params['M_hyst'] = partial(asoh.m_hyst.get_value, batch_id=batch_id)

        # Append the RC elements
        for r in range(params['num_RC_pairs']):
            params[f'R{r + 1}'] = partial(asoh.r[r + 1].get_value, batch_id=batch_id)
            params[f'C{r + 1}'] = partial(asoh.c[r].get_value, batch_id=batch_id)

        # Make the state
        state = TransientState(
            soc=transient.soc[batch_id % transient.soc.shape[0], 0],
            T_cell=transient.temp[batch_id % transient.temp.shape[0], 0],
            hyst=transient.hyst[batch_id % transient.hyst.shape[0], 0],
            eta_j=None if params['num_RC_pairs'] == 0 else transient.eta[batch_id % transient.eta.shape[0], :]
        )
        return params, state

    def update_transient_state(
            self,
            previous_inputs: TheveninInput,
            new_inputs: TheveninInput,
            transient_state: TheveninTransient,
            asoh: TheveninASOH
    ) -> TheveninTransient:
        # Return the current transient state if the time step is zero
        if np.isclose(new_inputs.time - previous_inputs.time, 0):
            return transient_state.model_copy(deep=True)

        # Initialize the array in which to store output values
        batch_size = max(transient_state.batch_size, asoh.batch_size, new_inputs.batch_size)
        output_array = np.zeros((batch_size, len(transient_state)))

        # Iterate over models representing each member of the batch
        for model_i, (model, state) in enumerate(self._make_models(transient_state, asoh, new_inputs)):
            # Propagate the system under a constant current load
            # TODO (wardlt): Make current time-dependent, which is possible by passing a callable
            cur_time = new_inputs.time[model_i % new_inputs.time.shape[0], 0]
            pre_time = previous_inputs.time[model_i % previous_inputs.time.shape[0], 0]
            current = -new_inputs.current[model_i % new_inputs.current.shape[0], 0]
            sln = model.take_step(state, current=current, delta_t=cur_time - pre_time)

            # Fill in the state variables
            output_array[model_i, 0] = sln.soc
            output_array[model_i, 1] = sln.T_cell
            output_array[model_i, 2] = sln.hyst
            if model.num_RC_pairs > 0:
                output_array[model_i, 3:] = sln.eta_j

        return transient_state.make_copy(output_array)

    def calculate_terminal_voltage(
            self,
            new_inputs: InputQuantities,
            transient_state: TheveninTransient,
            asoh: TheveninASOH) -> OutputQuantities:
        # Thevenin stores overpotentials, so it is easy enough to compute terminal voltage directly from states and ASOH
        #  See last eq of https://rovi-org.github.io/thevenin/user_guide/model_description.html
        v = (
                asoh.ocv.get_value(transient_state.soc)
                # Sign convention is opposite of thevenin
                + new_inputs.current[:, 0] * asoh.r[0].get_value(transient_state.soc, transient_state.temp)
                - transient_state.eta.sum(axis=1, keepdims=True)
        )
        return OutputQuantities(terminal_voltage=v)
