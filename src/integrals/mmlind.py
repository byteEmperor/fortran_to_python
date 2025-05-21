# src/integrals/mmlind.py

import numpy as np
from .utils import MACHINE_INF, MACHINE_MIN, ERR_TOL, get_argmin_argmax

def mmlind(x: float, y: float, z: float) -> tuple[float, int]:
    """
    Compute the incomplete elliptic integral of the second kind.

    Parameters:
    - x, y: non-negative floats
    - z: positive float

    Returns:
    - value: The computed integral value or np.inf on error
    - ier: Error code
        0 = OK
        129 = one of x, y, z is negative
        130 = x + y or z is less than argmin
        131 = x, y, or z is greater than argmax
    """

    ier = 0
    argmin, argmax = get_argmin_argmax()

    # Error checks
    if min(x, y, z) < 0:
        return MACHINE_INF, 129
    if x + y < argmin or z < argmin:
        return MACHINE_INF, 130
    if max(x, y, z) > argmax:
        return MACHINE_INF, 131

    xn, yn, zn = x, y, z
    sigma = 0.0
    power4 = 1.0

    # Iterative averaging loop
    while True:
        mu = (xn + yn + 3.0 * zn) * 0.2
        xndev = (mu - xn) / mu
        yndev = (mu - yn) / mu
        zndev = (mu - zn) / mu
        epslon = max(abs(xndev), abs(yndev), abs(zndev))

        if epslon < ERR_TOL:
            break

        xnroot = np.sqrt(xn)
        ynroot = np.sqrt(yn)
        znroot = np.sqrt(zn)

        lamda = xnroot * (ynroot + znroot) + ynroot * znroot
        sigma += power4 / (znroot * (zn + lamda))
        power4 *= 0.25
        xn = (xn + lamda) * 0.25
        yn = (yn + lamda) * 0.25
        zn = (zn + lamda) * 0.25

    # Correction terms
    c1, c2, c3, c4 = 3/14, 1/6, 9/22, 3/26
    ea = xndev * yndev
    eb = zndev**2
    ec = ea - eb
    ed = ea - 6 * eb
    ef = ed + 2 * ec

    s1 = ed * (-c1 + 0.25 * c3 * ed - 1.5 * c4 * zndev * ef)
    s2 = zndev * (c2 * ef + zndev * (-c3 * ec + zndev * c4 * ea))
    value = 3 * sigma + power4 * (1 + s1 + s2) / (mu * np.sqrt(mu))

    return value, ier
