# test/test_mmlind.py

import pytest
from src.integrals.mmlind import mmlind

@pytest.mark.parametrize(
    "x, y, z, expected_value, expected_ier",
    [
        # Actual values passed through mmlind from the fortran simulation
        (4.0155639732177946, 0.72507474213226519, 6.9123207026445911E-020, 6687205229.9860315, 0),
        (6.9123207112741198E-020, 0.72507474213226519, 4.0155639732177946, 0.51803648185906126, 0),
        (4.0155639732177946, 6.9123207112741198E-020, 0.72507474213226519, 1.8877806496346259, 0),
        (4.0155639732315986, 0.72507474193511645, 2.0522431956897008E-019, 3880986922.2885671, 0),
        (4.0155639732098152, 2.3425659624566667E-020, 0.72507474224619484, 1.8877806495890610, 0),
        (6.8328834667540349E-020, 0.72507474213383660, 4.0155639732176844, 0.51803648185941487, 0)

    ]
)
def test_mmlind_known_values(x, y, z, expected_value, expected_ier):
    value, ier = mmlind(x, y, z)
    assert ier == expected_ier, f"Expected ier={expected_ier}, got {ier}"
    if ier == 0:
        assert abs(value - expected_value) < 1e-2, f"Value off: {value} vs {expected_value}"

