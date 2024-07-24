from typing import Tuple

from pytest import fixture

from moirae.models.ecm import ECMASOH, ECMTransientVector, EquivalentCircuitModel


@fixture()
def rint_parameters() -> Tuple[ECMASOH, ECMTransientVector, EquivalentCircuitModel]:
    """Get the parameters which define an uncharged R_int model"""

    return (ECMASOH.provide_template(has_C0=False, num_RC=0, H0=0.0),
            ECMTransientVector.provide_template(has_C0=False, num_RC=0),
            EquivalentCircuitModel())
