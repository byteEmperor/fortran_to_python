# structure_formation.models.derivs_evol_delta.py

import numpy as np
from structure_formation.numerics.integrals.elliprj import mmlind_scipy

def derivs_evol_delta(t, y, simulation_params, time_params):
    """
    Ellipsoidal evolution with dynamically evolving delta(t)
    No tides, no Lambda.
    """

    a1, v1 = y[0], y[1]
    a2, v2 = y[2], y[3]
    a3, v3 = y[4], y[5]

    dydt = np.zeros_like(y)

    # Alpha_m from Carlson elliptic projection
    alpha = [0.0, 0.0, 0.0]
    alpha[0], _ = mmlind_scipy((a2 / a1)**2, (a3 / a1)**2, 1.0)
    alpha[1], _ = mmlind_scipy((a1 / a2)**2, (a3 / a2)**2, 1.0)
    alpha[2], _ = mmlind_scipy((a1 / a3)**2, (a2 / a3)**2, 1.0)

    if t == 0:
        t = 1e-10
    rho_u = 1.0 / t**2

    # === Dynamic delta ===
    vol_e = max(a1 * a2 * a3, 1e-8)
    delta_t = t ** 2 / vol_e - 1

    for i, (a, v, am) in enumerate(zip([a1, a2, a3], [v1, v2, v3], alpha)):
        dydt[2 * i] = v
        coeff = (1 + delta_t) / 3 + 0.5 * (am - 2/3) * delta_t
        acc = 4 * np.pi * rho_u * coeff * a
        dydt[2 * i + 1] = acc

    return dydt
