"""Interfaces that evaluate the fitness of a set of battery state parameters
provided as a NumPy array."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import rv_continuous
from battdat.data import BatteryDataset

from ._base import BaseLoss
from ._meta import AdditiveLoss
from moirae.interface import row_to_inputs
from moirae.models.base import CellModel, HealthVariable, GeneralContainer, InputQuantities, OutputQuantities
from moirae.simulator import Simulator
from moirae.models.ecm import ECMInput, ECMMeasurement


# TODO (wardlt): Generalize to other outputs when we have them
@dataclass
class MeanSquaredLoss(BaseLoss):
    """
    Score the fitness of a set of health parameters by the mean squared error
    between observed and predicted terminal voltage.
    """

    input_class: type[InputQuantities] = ECMInput
    """Class used to represent the input data for a model"""
    output_class: type[OutputQuantities] = ECMMeasurement
    """Class used to represent the output data for a model"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Translate input parameters to state and ASOH parameters
        state_x, asoh_x = self.x_to_state(x, inplace=True)

        # Build a simulator
        raw_data = self.observations.tables['raw_data']
        initial_input, initial_output = row_to_inputs(raw_data.iloc[0],
                                                      input_class=self.input_class,
                                                      output_class=self.output_class)
        sim = Simulator(
            cell_model=self.cell_model,
            asoh=asoh_x,
            transient_state=state_x,
            initial_input=initial_input,
        )

        # Prepare the output arrays
        num_outs = len(initial_output)
        pred_y = np.zeros((len(raw_data), 1, num_outs))
        true_y = np.zeros((len(raw_data), x.shape[0], num_outs))

        true_y[0, :] = initial_output.to_numpy()
        y = self.cell_model.calculate_terminal_voltage(initial_input, state_x, asoh_x)
        pred_y[0, :] = y.to_numpy()

        # Run the forward model
        for i, (_, row) in enumerate(self.observations.tables['raw_data'].iloc[1:].iterrows()):
            new_in, new_out = row_to_inputs(row,
                                            input_class=self.input_class,
                                            output_class=self.output_class)
            _, pred_out = sim.step(new_in)

            true_y[i + 1, :] = new_out.to_numpy()
            pred_y[i + 1, :] = pred_out.to_numpy()

        # Compute the mean-squared-error for each member of the batch
        squared_error = np.power(pred_y - true_y, 2)
        return np.mean(squared_error, axis=(0, 2))  # Average over steps and outputs


@dataclass
class PriorLoss(BaseLoss):
    """Compute the negative log-probability of parameter values from a prior distribution

    Supply priors as a scipy :class:`~scipy.stats.rv_continuous` distribution
    that defines the :meth:`~scipy.stats.rv_continuous.logpdf` method.

    For example, setting priors for the hysteresis parameter of an ECM
    and no priors for the ASOH parameters.

    .. code-block:: python

       from scipy.stats import norm

       hy_dist = norm(loc=0, scale=0.1)
       prior_loss = PriorLoss(
           transient_priors={'hyst': hy_dist},
           asoh_priors={},
           cell_model=ecm_model,
           asoh=init_asoh,
           transient_state=int_state,
           observations=timeseries_dataset
       )

    """

    def __init__(self,
                 transient_priors: dict[str, rv_continuous],
                 asoh_priors: dict[str, rv_continuous],
                 cell_model: CellModel,
                 asoh: HealthVariable,
                 transient_state: GeneralContainer,
                 observations: BatteryDataset):
        super().__init__(
            cell_model=cell_model,
            asoh=asoh,
            transient_state=transient_state,
            observations=observations
        )
        self._tran_priors = transient_priors.copy()
        self._asoh_priors = asoh_priors.copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Prepare the outputs
        state_x, asoh_x = self.x_to_state(x)
        output = np.zeros((x.shape[0],))

        # Sum up the log priors
        for name, dist in self._tran_priors.items():
            output -= dist.logpdf(getattr(state_x, name)).sum(axis=1)

        for name, dist in self._asoh_priors.items():
            output -= dist.logpdf(asoh_x.get_parameters([name])).sum(axis=1)

        return output


__all__ = [
    'BaseLoss', 'MeanSquaredLoss', 'PriorLoss', 'AdditiveLoss'
]
