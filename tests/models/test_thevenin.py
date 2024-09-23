"""Test the interface to the Thevenin package"""

from moirae.models.thevenin.components import SOCPolynomialVariable, SOCTempPolynomialVariable
from moirae.models.thevenin.state import TheveninASOH

rint = TheveninASOH(
    capacity=1.,
    ocv=SOCPolynomialVariable(coeffs=[1.5, 1.]),
    r=[SOCTempPolynomialVariable(soc_coeffs=[0.01, 0.01], t_coeffs=[0, 0.001])]
)


def test_rint():
    """Ensure the r'int model's subcomponents work"""

    # Ensuring we can do SOC dependence
    assert rint.ocv(0.5) == 2.

    # Ensuring we can do SOC and temperature dependence
    assert rint.r[0](0.5, 298) == 0.015
    assert rint.r[0](0.5, 308) == 0.025
