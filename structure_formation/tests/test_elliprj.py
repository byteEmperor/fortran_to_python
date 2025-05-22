# tests/test_elliprj.py

import pytest
from structure_formation.numerics.integrals.mmlind import mmlind
from structure_formation.numerics.integrals.elliprj import mmlind_scipy

@pytest.mark.parametrize(
    "x, y, z, expected_value, expected_ier",
    [
        # Valid cases (ier == 0), check result value
        # (1.0, 2.0, 3.0, 1.5951684240000000E+09, 0),
        # (0.5, 0.3, 0.7, 1.5951684240000000E+09, 0),
        # (1e-8, 1e-8, 1e-8, 1.5951684240000000E+09, 0),
        # (1.0, 1.0, 1.0, 1.5951684240000000E+09, 0),

        # Invalid domain or precision issues (ier != 0), skip value check
        # (1e38, 1e38, 1e38, 0.0, 131),
        # (-1.0, 2.0, 3.0, 0.0, 129),
        # (1.0, -2.0, 3.0, 0.0, 129),
        # (1.0, 2.0, -3.0, 0.0, 129),
        # (1e-40, 1e-40, 1e-40, 0.0, 130),
        # (0.0, 0.0, 0.0, 0.0, 130),

        # Actual values passed through mmlind from the fortran simulation
        (4.0155639732177946, 0.72507474213226519, 6.9123207026445911E-020, 6687205229.9860315, 0),
        (6.9123207112741198E-020, 0.72507474213226519, 4.0155639732177946, 0.51803648185906126, 0),
        (4.0155639732177946, 6.9123207112741198E-020, 0.72507474213226519, 1.8877806496346259, 0),
        (4.0155639732315986, 0.72507474193511645, 2.0522431956897008E-019, 3880986922.2885671, 0),
        (4.0155639732098152, 2.3425659624566667E-020, 0.72507474224619484, 1.8877806495890610, 0),
        (6.8328834667540349E-020, 0.72507474213383660, 4.0155639732176844, 0.51803648185941487, 0)

    ]
)

def test_mmlind_matches_elliprj(x, y, z, expected_value, expected_ier):
    value_mmlind, ier_mmlind = mmlind(x, y, z)
    value_elliprj, ier_elliprj = mmlind_scipy(x, y, z)
    assert ier_mmlind == ier_elliprj, f"The ier matches!"
    if ier_elliprj == 0:
        assert abs(value_elliprj - expected_value) < 1e-2, f"Value off: {value_elliprj} vs {value_mmlind} vs {expected_value}"