import numpy as np
from structure_formation.numerics.integrals.elliprj import mmlind_scipy

import numpy as np

def sphericalizing_derivs_dark(t, y, simulation_params, time_params):
    """
    Custom derivs function to model ellipsoidal void evolution with:
    - EdS behavior for delta = 0: a(t) ∝ t^(2/3), d²a/dt² ∝ t^(-4/3)
    - Axis ratios → 1 for underdense regions
    - Includes dark energy (Λ) term

    Inputs:
    - t: time (float)
    - y: state vector: [a1, v1, a2, v2, a3, v3]
    - simulation_params: has delta and optional Lambda attribute
    - time_params: (not used but accepted)

    Returns:
    - dydt: array of time derivatives
    """
    delta = simulation_params.delta
    Lambda = simulation_params.lambda0

    # Unpack axes and velocities
    a1, v1 = y[0], y[1]
    a2, v2 = y[2], y[3]
    a3, v3 = y[4], y[5]

    dydt = np.zeros_like(y)

    # === Velocity updates (positions) ===
    dydt[0] = v1
    dydt[2] = v2
    dydt[4] = v3

    # === Reference EdS behavior for delta = 0 ===
    if t == 0:
        t = 1e-10  # avoid division by zero

    if abs(delta) < 1e-8:
        for i, a in enumerate([a1, a2, a3]):
            dydt[2*i + 1] = -2 / (9 * t**(4/3)) * a + Lambda * a
        return dydt

    # === For underdensities (delta < 0), force spherical expansion ===
    axes = np.array([a1, a2, a3])
    velocities = np.array([v1, v2, v3])
    mean_a = np.mean(axes)

    for i in range(3):
        a = axes[i]
        v = velocities[i]
        damping = -0.5 * delta * (a - mean_a) / max(t, 1e-5)
        acc = max(-2 / (9 * t**(4/3)) + damping, 0.0) + Lambda * a
        dydt[2*i + 1] = acc
        dydt[2*i] = max(v, 0.0)

    return dydt


