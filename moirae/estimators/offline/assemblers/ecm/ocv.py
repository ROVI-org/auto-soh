"""
OCV assembler
"""
from typing import Union
from typing_extensions import Self

import numpy as np

from moirae.models.ecm.components import OpenCircuitVoltage as OCV
from moirae.estimators.offline.extractors.base import ExtractedParameter
from moirae.estimators.offline.extractors.ecm.ocv import ExtractedOCV
from moirae.estimators.offline.assemblers.utils import SOCRegressor

from .utils import SOCDependentAssembler


class OCVAssembler(SOCDependentAssembler):
    """
    Assembles an OCV object from extracted parameters, which always employs an Isotonic regression.

    Args:
        soc_points: points in the SOC domain to be used when assembling the OCV object
    """
    def __init__(self,
                 soc_points: np.ndarray = np.linspace(0, 1, 11)):
        super().__init__(regressor=SOCRegressor(style='isotonic', parameters={'out_of_bounds': 'clip'}),
                         soc_points=soc_points)

    def _prepare_for_regression(self, extracted_parameter: Union[ExtractedOCV, ExtractedParameter]):
        # Invoke the base capabilities of the parent class for cleaning up
        clean_up = super()._prepare_for_regression(extracted_parameter=extracted_parameter)

        # If we provided the currents, we need to clean that up as well
        if 'current' in clean_up.keys():
            # Weights should be inversely proportional to current: larger current => smaller weight
            inv_curr = 1.0 / abs(np.array(clean_up.pop('current')))
            inv_curr = inv_curr / sum(inv_curr)
            clean_up['sample_weight'] = inv_curr

        return clean_up

    def assemble(self, extracted_parameter: Union[ExtractedOCV, ExtractedParameter]) -> OCV:
        if extracted_parameter['units'] != 'Volt':
            unit = extracted_parameter['units']
            raise ValueError(f'OCV provided in {unit}, rather than Volt!')

        # Get preliminary OCV, but as a different object
        ocv_ref =  super().assemble(extracted_parameter=extracted_parameter)

        return OCV(ocv_ref=ocv_ref)
