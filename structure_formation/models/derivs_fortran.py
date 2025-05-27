# structure_formation/models/derivs_fortran.py

import numpy as np
from structure_formation.numerics.integrals.elliprj import mmlind_scipy

# def derivs_fortran_style(t, y, simulation_params, time_params):
#     """
#     Derivatives for ellipsoidal evolution using a constant initial density contrast δ₀,
#     and fixed gravitational anisotropy via αᵢ. This implementation follows the classical
#     ellipsoidal model from Bond & Myers and van de Weygaert.
#
#     No δ(t) feedback, no shape damping, no oscillatory driving.
#     """
#
#     a1, v1 = y[0], y[1]
#     a2, v2 = y[2], y[3]
#     a3, v3 = y[4], y[5]
#     dydt = np.zeros_like(y)
#
#     # Prevent division by zero
#     if t == 0:
#         t = 1e-10
#
#     # Background density (EdS): ρ_u ∝ t⁻²
#     rho_u = 1.0 / t**2
#     delta0 = simulation_params.delta
#
#     # Shape-dependent αᵢ from elliptic integrals
#     alpha = [0.0, 0.0, 0.0]
#     alpha[0], _ = mmlind_scipy((a2 / a1)**2, (a3 / a1)**2, 1.0)
#     alpha[1], _ = mmlind_scipy((a1 / a2)**2, (a3 / a2)**2, 1.0)
#     alpha[2], _ = mmlind_scipy((a1 / a3)**2, (a2 / a3)**2, 1.0)
#
#     for i, (a, v, alpha_i) in enumerate(zip([a1, a2, a3], [v1, v2, v3], alpha)):
#         dydt[2 * i] = v
#         alpha_i = max(alpha_i, 0.05)
#         if delta0 < 0 and alpha_i < 0:
#             print(f"[WARNING] Negative alpha_i = {alpha_i:.4f} at t = {t:.4f}, axis = {i}")
#
#         alpha_i = max(alpha_i, 1e-3)
#
#         # Classical ellipsoidal model acceleration
#         # coeff = delta0 / 3.0 + 0.5 * (alpha_i - 2.0 / 3.0) * delta0
#         # acc = -4.0 * np.pi * rho_u * a * coeff
#         # Clean form: ä = -4πGρ a × (δ₀ × αᵢ / 2)
#         coeff = delta0 * alpha_i / 2.0
#         acc = -4 * np.pi * rho_u * a * coeff
#
#         dydt[2 * i + 1] = acc
#
#     return dydt

# This kinda works
def derivs_fortran_style(t, y, simulation_params, time_params):
    """
    Hybrid ellipsoidal evolution model:
    - For δ₀ > 0: conservative collapse with fixed δ₀ and αᵢ.
    - For δ₀ < 0: void expansion with δ(t) feedback, damping, and shape restoring force.

    Prevents oscillations and runaway axis divergence.
    """

    # Unpack axis lengths and velocities
    a1, v1 = y[0], y[1]
    a2, v2 = y[2], y[3]
    a3, v3 = y[4], y[5]
    dydt = np.zeros_like(y)

    if t == 0:
        t = 1e-10  # prevent division by zero

    # Elliptic integral alpha_i
    alpha = [0.0, 0.0, 0.0]
    alpha[0], _ = mmlind_scipy((a2 / a1)**2, (a3 / a1)**2, 1.0)
    alpha[1], _ = mmlind_scipy((a1 / a2)**2, (a3 / a2)**2, 1.0)
    alpha[2], _ = mmlind_scipy((a1 / a3)**2, (a2 / a3)**2, 1.0)

    # Background density in EdS: ρ_u ∝ t⁻²
    rho_u = 1.0 / t**2
    delta0 = simulation_params.delta
    vol_e = max(a1 * a2 * a3, 1e-8)
    delta_t = t**2 / vol_e - 1.0

    a_list = [a1, a2, a3]
    v_list = [v1, v2, v3]

    # === OVERDENSITY: Conservative collapse ===
    if delta0 >= 0:
        for i, (a, v, alpha_i) in enumerate(zip(a_list, v_list, alpha)):
            dydt[2 * i] = v
            coeff = delta0 / 3.0 + 0.5 * (alpha_i - 2.0 / 3.0) * delta0
            acc = -4.0 * np.pi * rho_u * a * coeff
            dydt[2 * i + 1] = acc

    # === UNDERDENSITY: Feedback + damping + restoring ===
    else:
        axes = np.array([a1, a2, a3])
        velocities = np.array([v1, v2, v3])
        mean_a = np.mean(axes)

        # Encourage axis equalization and outward expansion
        for i in range(3):
            a = axes[i]
            v = velocities[i]
            damping = -0.5 * delta0 * (a - mean_a) / max(t, 1e-5)
            acc = max(-2 / (9 * t ** (4 / 3)) + damping, 0.0)
            dydt[2 * i + 1] = acc  # Ensure no negative acceleration
            dydt[2 * i] = max(v, 0.0)  # Prevent velocity from becoming negative

    return dydt

# def derivs_fortran_style(t, y, simulation_params, time_params):
#     """
#     Hybrid ellipsoidal evolution model:
#     - For δ₀ > 0: conservative collapse with fixed δ₀ and αᵢ.
#     - For δ₀ < 0: void expansion with δ(t) feedback, damping, and shape restoring force.
#
#     Prevents oscillations and runaway axis divergence.
#     """
#
#     # Unpack axis lengths and velocities
#     a1, v1 = y[0], y[1]
#     a2, v2 = y[2], y[3]
#     a3, v3 = y[4], y[5]
#     dydt = np.zeros_like(y)
#
#     if t == 0:
#         t = 1e-10  # prevent division by zero
#
#     # Elliptic integral alpha_i
#     alpha = [0.0, 0.0, 0.0]
#     alpha[0], _ = mmlind_scipy((a2 / a1)**2, (a3 / a1)**2, 1.0)
#     alpha[1], _ = mmlind_scipy((a1 / a2)**2, (a3 / a2)**2, 1.0)
#     alpha[2], _ = mmlind_scipy((a1 / a3)**2, (a2 / a3)**2, 1.0)
#
#     # Background density in EdS: ρ_u ∝ t⁻²
#     rho_u = 1.0 / t**2
#     delta0 = simulation_params.delta
#     vol_e = max(a1 * a2 * a3, 1e-8)
#     delta_t = t**2 / vol_e - 1.0
#
#     a_list = [a1, a2, a3]
#     v_list = [v1, v2, v3]
#
#     # === OVERDENSITY: Conservative collapse ===
#     if delta0 >= 0:
#         for i, (a, v, alpha_i) in enumerate(zip(a_list, v_list, alpha)):
#             dydt[2 * i] = v
#             coeff = delta0 / 3.0 + 0.5 * (alpha_i - 2.0 / 3.0) * delta0
#             acc = -4.0 * np.pi * rho_u * a * coeff
#             dydt[2 * i + 1] = acc
#
#     # === UNDERDENSITY: Feedback + damping + restoring ===
#     else:
#         # Parameters
#         gamma = 0.6  # shape strength
#         beta = 0.5  # velocity damping
#         kappa = 0.1  # restoring strength
#
#         a_mean = sum(a_list) / 3.0
#
#         for i, (a, v, alpha_i) in enumerate(zip(a_list, v_list, alpha)):
#             dydt[2 * i] = v
#
#             scale = np.exp(- (a / a_mean - 1.0))
#
#             shape_correction = gamma * (alpha_i - 1.0 / 3.0) * delta_t * scale
#             base_term = (1 + delta_t) / 3.0
#             coeff = max(base_term + shape_correction, 0.0)
#
#             acc_gravity = 4.0 * np.pi * rho_u * a * coeff
#             damping_term = -beta * v * scale
#             restoring_term = kappa * (a_mean - a) * scale
#
#             acc = acc_gravity + damping_term + restoring_term
#             deviation = (a - a_mean) / a_mean
#             if deviation > 0:
#                 acc *= 1.0 / (1.0 + 3.0 * deviation ** 2)
#             dydt[2 * i + 1] = acc
#
#     return dydt

