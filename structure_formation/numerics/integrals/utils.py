# structure_formation/integrals/utils.py

import numpy as np

# Equivalent to the Fortran values
MACHINE_INF = 1.7e38
MACHINE_MIN = 2.938735878e-39
ERR_TOL = 1e-10


def get_argmin_argmax(p: int = 15) -> tuple[float, float]:
    """
    Calculate argmin and argmax based on machine limits and precision.

    p: number of decimal digits of precision (default ~15 for double-precision)

    Returns:
    - (argmin, argmax)
    """
    argmn1 = 3.0 * (MACHINE_MIN ** (2.0 / 3.0))
    argmn2 = 3.0 / (MACHINE_INF ** (2.0 / 3.0))
    argmin = max(argmn1, argmn2)

    argmax = ((0.085 * (p) ** (-1.0 / 6.0)) / MACHINE_MIN) ** (2.0 / 3.0)

    return argmin, argmax
