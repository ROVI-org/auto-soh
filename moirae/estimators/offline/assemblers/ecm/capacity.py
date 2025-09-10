"""
Capacity assembler
"""
from moirae.models.ecm.components import MaxTheoreticalCapacity
from moirae.estimators.offline.extractors.base import ExtractedParameter
from moirae.estimators.offline.assemblers.base import BaseAssembler


class CapacityAssembler(BaseAssembler):
    """
    Assembles capacity object
    """
    def __init__(self):
        pass

    def assemble(self, extracted_parameter: ExtractedParameter) -> MaxTheoreticalCapacity:
        if extracted_parameter['units'] != 'Amp-hour':
            unit = extracted_parameter['units']
            raise ValueError(f'Capacity provided in {unit}, rather than Amp-hour!')

        value = extracted_parameter['value']
        if value < 0:
            raise ValueError(f'Negative capacity {value:.2f} provided!')

        return MaxTheoreticalCapacity(base_values=value)
