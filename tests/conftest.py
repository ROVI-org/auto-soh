from typing import Tuple

from pytest import fixture

from moirae.models.ecm import ECMASOH, ECMTransientVector, EquivalentCircuitModel, ECMInput


@fixture()
def simple_rint() -> Tuple[ECMASOH, ECMTransientVector, ECMInput, EquivalentCircuitModel]:
    """Get the parameters which define an uncharged R_int model"""

    return (ECMASOH.provide_template(has_C0=False, num_RC=0, H0=0.0),
            ECMTransientVector.provide_template(has_C0=False, num_RC=0),
            ECMInput(time=0., current=0.),
            EquivalentCircuitModel())
