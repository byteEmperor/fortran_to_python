# structure_formation/models/derivs.py

import math
import numpy as np
from scipy.optimize import brentq

from .zeropar_functions import openU, closedU
from structure_formation.numerics.roots.zbrent import zbrent
from structure_formation.numerics.integrals.mmlind import mmlind
from structure_formation.numerics.integrals.elliprj import mmlind_scipy

from structure_formation.simulation.simulation_parameters import SimulationParameters
from structure_formation.models.time_utils import TimeBounds

def derivs(t, y, simulation_params: SimulationParameters, time_params: TimeBounds):
    # Unpack parameters
    e = simulation_params.e
    delta = simulation_params.delta
    Omega0 = simulation_params.Omega0
    ai = simulation_params.ai
    zi = simulation_params.zi
    factor = simulation_params.factor
    aEnd = simulation_params.aEnd
    e11, e22, e33 = simulation_params.e11, simulation_params.e22, simulation_params.e33

    sigma = time_params.sigma

    # Fortran used COMMON to make ztau global — we'll just define it locally
    ztau = t

    # Wrapper functions with sigma and ztau bound in
    def openU_wrapped(theta):
        return openU(theta, sigma, ztau)

    def closedU_wrapped(theta):
        return closedU(theta, sigma, ztau)

    # Cosmological scale factor aexp from development angle
    if abs(Omega0 - 1.0) < 1e-8:
        aexp = (3.0 ** (1.0 / 3.0)) * (t ** (2.0 / 3.0)) / (1.0 + zi)
    elif Omega0 < 1.0:
        psi = zbrent(openU_wrapped, 0.0, 20.0, 1e-6)
        aexp = (math.cosh(psi) - 1.0) * Omega0 / (2.0 * (1.0 - Omega0))
    else:
        thet = zbrent(closedU_wrapped, -5.0, 5.0, 1e-6)
        aexp = (1.0 - math.cos(thet)) * Omega0 / (2.0 * (Omega0 - 1.0))

    # if abs(Omega0 - 1.0) < 1e-8:
    #     aexp = (3.0 ** (1.0 / 3.0)) * (t ** (2.0 / 3.0)) / (1.0 + zi)
    # elif Omega0 < 1.0:
    #     psi = brentq(openU_wrapped, 0.0, 20.0, xtol=1e-6)
    #     aexp = (math.cosh(psi) - 1.0) * Omega0 / (2.0 * (1.0 - Omega0))
    # else:
    #     thet = brentq(closedU_wrapped, -5.0, 5.0, xtol=1e-6)
    #     aexp = (1.0 - math.cos(thet)) * Omega0 / (2.0 * (Omega0 - 1.0))

    auniv = aexp
    aexp = ai / aexp
    twothr = 2.0 / 3.0

    # Axis stretches
    a1, a2, a3 = y[0], y[2], y[4]
    b1 = e[0] * a1
    b2 = e[1] * a2
    b3 = e[2] * a3
    b12, b22, b32 = b1**2, b2**2, b3**2

    # mmlind returns a tuple, so we take the first value
    alpha = [0, 0, 0]
    alpha[0], _ = mmlind(b32, b22, b12)
    alpha[1], _ = mmlind(b12, b32, b22)
    alpha[2], _ = mmlind(b12, b22, b32)
    alpha = [twothr * b1 * b2 * b3 * a for a in alpha]

    # External tidal evolution
    e11t = auniv * e11
    e22t = auniv * e22
    e33t = auniv * e33

    dydt = np.zeros(6)
    dydt[0] = y[1]
    #dydt[1] = -delta * alpha[0] / (y[2] * y[4]) - (aexp**3)*(twothr - alpha[0] + e11t)*y[0]
    #dydt[1] = -delta * alpha[0] / (y[2] * y[4]) - (aexp ** 3) * (twothr - alpha[0]) * y[0] - (aexp ** 3) * e11t * y[0]
    dydt[1] = -delta * alpha[0] / (y[2] * y[4]) - (aexp ** 3) * (alpha[0] + e11t) * y[0]
    dydt[2] = y[3]
    #dydt[3] = -delta * alpha[1] / (y[0] * y[4]) - (aexp**3)*(twothr - alpha[1] + e22t)*y[2]
    #dydt[3] = -delta * alpha[1] / (y[0] * y[4]) - (aexp ** 3) * (twothr - alpha[1]) * y[2] - (aexp ** 3) * e22t * y[2]
    dydt[3] = -delta * alpha[1] / (y[0] * y[4]) - (aexp ** 3) * (alpha[1] + e22t) * y[2]
    dydt[4] = y[5]
    #dydt[5] = -delta * alpha[2] / (y[0] * y[2]) - (aexp**3)*(twothr - alpha[2] + e33t)*y[4]
    #dydt[5] = -delta * alpha[2] / (y[0] * y[2]) - (aexp ** 3) * (twothr - alpha[2]) * y[4] - (aexp ** 3) * e33t * y[4]
    dydt[5] = -delta * alpha[2] / (y[0] * y[2]) - (aexp ** 3) * (alpha[2] + e33t) * y[4]

    return dydt
