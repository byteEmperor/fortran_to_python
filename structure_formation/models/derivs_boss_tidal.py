import numpy as np
#from structure_formation.numerics.integrals.mmlind import mmlind_scipy  # if available
from structure_formation.numerics.integrals.mmlind import mmlind



def derivs_boss_tidal(t, y, simulation_params, time_params):
    """
    Accurate ellipsoidal evolution including anisotropic tidal fields.

    Parameters:
    - simulation_params must have:
        .delta (float): initial density contrast
        .tau (array-like): tidal eigenvalues [tau1, tau2, tau3]

    Returns:
    - dydt: time derivatives of [a1, v1, a2, v2, a3, v3]
    """
    delta0 = simulation_params.delta
    tau = getattr(simulation_params, 'tau', [0.0, 0.0, 0.0])  # τ1, τ2, τ3

    # Unpack axes and velocities
    a1, v1 = y[0], y[1]
    a2, v2 = y[2], y[3]
    a3, v3 = y[4], y[5]
    axes = np.array([a1, a2, a3])
    velocities = np.array([v1, v2, v3])
    dydt = np.zeros_like(y)

    if t <= 0:
        t = 1e-10

    # Volume and evolving delta(t)
    vol_e = max(a1 * a2 * a3, 1e-8)
    delta_t = t**2 / vol_e - 1.0

    # Elliptic integral terms α_i (symmetry-preserving)
    alpha = [0.0, 0.0, 0.0]
    alpha[0], _ = mmlind((a2 / a1)**2, (a3 / a1)**2, 1.0)
    alpha[1], _ = mmlind((a1 / a2)**2, (a3 / a2)**2, 1.0)
    alpha[2], _ = mmlind((a1 / a3)**2, (a2 / a3)**2, 1.0)

    # Background matter density (EdS): ρ_u ∝ t⁻²
    rho_u = 1.0 / t**2
    G = 1.0  # Normalized units

    for i in range(3):
        a = axes[i]
        v = velocities[i]
        alpha_i = alpha[i]
        tau_i = tau[i]

        # Velocity
        dydt[2*i] = v

        # Acceleration: gravity + anisotropic term + tidal term
        grav_term = -4 * np.pi * G * rho_u * a * (delta_t / 3 + 0.5 * (alpha_i - 2/3) * delta_t)
        tidal_term = -tau_i * a

        # Total acceleration
        dydt[2*i + 1] = grav_term + tidal_term

    return dydt
