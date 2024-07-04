from pydantic import Field
from asoh.models.base import InputQuantities


class ECMInput(InputQuantities):
    """
    Control of a battery based on the feed current, temperature
    """
    temperature: float = Field(default=None,
                               description='Temperature reading(s). Units: °C')
