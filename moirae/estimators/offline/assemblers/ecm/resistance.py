"""
Resistance assembler
"""
import numpy as np

from moirae.estimators.offline.extractors.base import ExtractedParameter
from moirae.models.ecm.components import Resistance

from .utils import SOCDependentAssembler


class ResistanceAssembler(SOCDependentAssembler):
    """
    Assembles resistance object
    """
    def assemble(self, extracted_parameter: ExtractedParameter):
        # Check units
        if extracted_parameter['units'] != 'Ohm':
            unit = extracted_parameter['units']
            raise ValueError(f'Resistance provided in {unit}, rather than Ohm!')
        # Get base interpolation
        soc_interp = super().assemble(extracted_parameter=extracted_parameter)
        # Get base values
        base_vals = soc_interp.base_values.flatten()
        if np.any(base_vals <= 0.):
            raise ValueError(f'Non-positive resistance detected! Base values = {base_vals}!')

        return Resistance(base_values=base_vals, soc_pinpoints=soc_interp.soc_pinpoints.flatten())