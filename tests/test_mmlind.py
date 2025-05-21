# tests/test_mmlind.py

import pytest
from src.integrals.mmlind import mmlind

@pytest.mark.parametrize(
    "x, y, z, expected_value, expected_ier",
    [
        (1.0, 2.0, 3.0, 10.000000000000000, 0),
        (2.0, 1.0, 3.0, 10.000000000000000, 0),
        ()

    ]
)
def test_mmlind_known_values(x, y, z, expected_value, expected_ier):
    value, ier = mmlind(x, y, z)
    assert ier == expected_ier, f"Expected ier={expected_ier}, got {ier}"
    if ier == 0:
        assert abs(value - expected_value) < 1e-8, f"Value off: {value} vs {expected_value}"

