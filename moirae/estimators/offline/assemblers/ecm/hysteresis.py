"""
Defines hysteresis assembler
"""
from typing import Union

import numpy as np

from moirae.models.ecm.components import HysteresisParameters
from moirae.estimators.offline.extractors.base import ExtractedParameter
from moirae.estimators.offline.extractors.ecm.hysteresis import ExtractedHysteresis

from .utils import SOCDependentAssembler


class HysteresisAssembler(SOCDependentAssembler):

    def _compute_weights(self, extracted_parameter: Union[ExtractedHysteresis, ExtractedParameter]) -> np.ndarray:
        # Weights are computed in based on the amount of time since the (dis/)charging step began
        if 'ajdusted_curr' not in extracted_parameter.keys():
            return super()._compute_weights(extracted_parameter)

        # The instantaneous hysteresis approaches its limit as (1 - exp(-kappa * dt))
        weights = np.array(extracted_parameter['step_time'])
        return weights

    def _prepare_for_regression(self, extracted_parameter: Union[ExtractedHysteresis, ExtractedParameter]):
        # Invoke the base capabilities of the parent class for cleaning up
        clean_up = super()._prepare_for_regression(extracted_parameter=extracted_parameter)
        if 'step_time' in clean_up.keys():
            _ = clean_up.pop('step_time')
        if 'adjusted_curr' in clean_up.keys():
            _ = clean_up.pop('adjusted_curr')

        # Make sure we prepare the weights adequately
        if 'weights' in clean_up.keys():
            if self.regressor.style == 'isotonic':
                clean_up['sample_weight'] = clean_up.pop('weights')
            elif self.regressor.style == 'interpolate':
                _ = clean_up.pop('weights')
            elif (self.regressor.style == 'lsq') or (self.regressor.style == 'smooth'):
                clean_up['w'] = clean_up.pop('weights')

        return clean_up

    def assemble(self, extracted_parameter: Union[ExtractedHysteresis, ExtractedParameter]) -> HysteresisParameters:
        if extracted_parameter['units'] != 'Volt':
            unit = extracted_parameter['units']
            raise ValueError(f'OCV provided in {unit}, rather than Volt!')

        # Let's also make sure the signs of the values are correct
        param_sign = extracted_parameter.copy()
        param_sign['value'] = np.abs(param_sign['value'])

        # Get preliminary OCV, but as a different object
        h0 = super().assemble(extracted_parameter=param_sign)

        return HysteresisParameters(base_values=h0.base_values.flatten(),
                                    soc_pinpoints=h0.soc_pinpoints.flatten())
