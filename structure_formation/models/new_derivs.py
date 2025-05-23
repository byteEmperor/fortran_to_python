# structure_formation/models/new_derivs.py

import numpy as np
from structure_formation.numerics.integrals.mmlind import mmlind
from structure_formation.numerics.integrals.elliprj import mmlind_scipy

def new_derivs(t, y, simulation_params, time_params):
    """
    Computes dy/dt for a collapsing ellipsoidal region without tidal forces or Lambda.
    This version preserves symmetry when delta = 0.

    Parameters:
    - t: conformal time (float)
    - y: state vector [a1, v1, a2, v2, a3, v3]
    - simulation_params: SimulationParameters
    - time_params: TimeBounds (unused but passed for compatibility)

    Returns:
    - dydt: time derivatives of y
    """
    delta = simulation_params.delta

    a1, v1 = y[0], y[1]
    a2, v2 = y[2], y[3]
    a3, v3 = y[4], y[5]

    dydt = np.zeros_like(y)

    # Alpha_m values via Carlson elliptic integrals (symmetry-preserving)
    b1, b2, b3 = a1, a2, a3
    alpha = [0.0, 0.0, 0.0]
    alpha[0], _ = mmlind_scipy((b2 / b1)**2, (b3 / b1)**2, 1.0)
    alpha[1], _ = mmlind_scipy((b1 / b2)**2, (b3 / b2)**2, 1.0)
    alpha[2], _ = mmlind_scipy((b1 / b3)**2, (b2 / b3)**2, 1.0)

    # Background matter density: rho_u ∝ 1 / t^2 (EdS units)
    if t == 0:
        t = 1e-10  # prevent singularity
    rho_u = 1.0 / t**2

    for i, (a, v, am) in enumerate(zip([a1, a2, a3], [v1, v2, v3], alpha)):
        dydt[2 * i] = v
        coeff = delta / 3 + 0.5 * (am - 2/3) * delta
        dydt[2 * i + 1] = -4 * np.pi * rho_u * coeff * a

    return dydt

