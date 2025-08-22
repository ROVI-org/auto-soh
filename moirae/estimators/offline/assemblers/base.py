from moirae.models.base import HealthVariable
from moirae.estimators.offline.extractors.base import ExtractedParameter


class BaseAssembler():
    """
    Base definition for an assembler
    """
    def assemble(self, extracted_parameter: ExtractedParameter) -> HealthVariable:
        """
        Assembles the `HealthVariable` from the extracted information

        Args:
            extracted_parameter: extracted information about the parameter at hand, obtained from an extractor

        Returns:
            corresponding health variable
        """
        raise NotImplementedError("Please implement in child classes!")