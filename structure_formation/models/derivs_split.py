# structure_formation/models/derivs_split.py

import numpy as np
from structure_formation.numerics.integrals.elliprj import mmlind_scipy

def derivs_evol_delta_split(t, y, simulation_params, time_params):
    """
    Unified ellipsoidal evolution for both voids (δ < 0) and overdensities (δ ≥ 0).
    Behavior splits internally for physical accuracy.
    """

    a1, v1 = y[0], y[1]
    a2, v2 = y[2], y[3]
    a3, v3 = y[4], y[5]
    dydt = np.zeros_like(y)

    # Elliptic integrals
    alpha = [0.0, 0.0, 0.0]
    alpha[0], _ = mmlind_scipy((a2 / a1)**2, (a3 / a1)**2, 1.0)
    alpha[1], _ = mmlind_scipy((a1 / a2)**2, (a3 / a2)**2, 1.0)
    alpha[2], _ = mmlind_scipy((a1 / a3)**2, (a2 / a3)**2, 1.0)

    if t == 0:
        t = 1e-10

    vol_e = max(a1 * a2 * a3, 1e-8)
    rho_u = 1.0 / t**2
    delta_t = t**2 / vol_e - 1.0
    delta0 = simulation_params.delta

    a_list = [a1, a2, a3]
    v_list = [v1, v2, v3]

    if delta0 <= 0:
        # === VOID EVOLUTION ===
        decay = 1.0 / (1.0 + t)**0.5
        gamma = 1.3
        beta = 0.33

        a_mean = sum(a_list) / 3.0

        for i in range(3):
            a = a_list[i]
            v = v_list[i]
            alpha_i = alpha[i]

            dydt[2 * i] = v
            base = (1 + delta_t) / 3.0
            shape_correction = gamma * decay * (alpha_i - 1.0 / 3.0) * delta_t
            iso_damping = -beta * (a - a_mean)

            acc = 4.0 * np.pi * rho_u * a * (base + shape_correction) + iso_damping
            dydt[2 * i + 1] = acc

    else:
        # === COLLAPSE EVOLUTION ===
        for i in range(3):
            a = a_list[i]
            v = v_list[i]
            alpha_i = alpha[i]

            dydt[2 * i] = v
            coeff = delta0 / 3.0 + 0.5 * (alpha_i - 2.0 / 3.0) * delta0
            acc = -4.0 * np.pi * rho_u * a * coeff
            dydt[2 * i + 1] = acc

    return dydt

# def derivs_evol_delta_split(t, y, simulation_params, time_params):
#     """
#     Unified ellipsoidal evolution for both voids (δ < 0) and overdensities (δ ≥ 0).
#     Behavior splits internally for physical accuracy.
#     """
#
#     a1, v1 = y[0], y[1]
#     a2, v2 = y[2], y[3]
#     a3, v3 = y[4], y[5]
#     dydt = np.zeros_like(y)
#
#     # Elliptic integrals
#     alpha = [0.0, 0.0, 0.0]
#     alpha[0], _ = mmlind_scipy((a2 / a1)**2, (a3 / a1)**2, 1.0)
#     alpha[1], _ = mmlind_scipy((a1 / a2)**2, (a3 / a2)**2, 1.0)
#     alpha[2], _ = mmlind_scipy((a1 / a3)**2, (a2 / a3)**2, 1.0)
#
#     if t == 0:
#         t = 1e-10
#
#     vol_e = max(a1 * a2 * a3, 1e-8)
#     rho_u = 1.0 / t**2
#     delta_t = t**2 / vol_e - 1.0
#     delta0 = simulation_params.delta
#
#     a_list = [a1, a2, a3]
#     v_list = [v1, v2, v3]
#
#     if delta0 < 0:
#         # === VOID EVOLUTION ===
#         decay = 1.0 / (1.0 + t)**0.5
#         gamma = 1.3
#         beta = 0.33
#
#         a_mean = sum(a_list) / 3.0
#
#         for i in range(3):
#             a = a_list[i]
#             v = v_list[i]
#             alpha_i = alpha[i]
#
#             # Tides
#             E = [simulation_params.e11, simulation_params.e22, simulation_params.e33]
#             aexp = time_params.get_aexp(t) if hasattr(time_params, 'get_aexp') else None
#             if aexp is None:
#                 aexp = t ** (2 / 3) / (1 + simulation_params.zi)  # fallback for EdS
#             tidal_term = a * aexp * E[i]
#
#             dydt[2 * i] = v
#             base = (1 + delta_t) / 3.0
#             shape_correction = gamma * decay * (alpha_i - 1.0 / 3.0) * delta_t
#             iso_damping = -beta * (a - a_mean)
#
#             acc = 4.0 * np.pi * rho_u * a * (base + shape_correction) + iso_damping + tidal_term
#             dydt[2 * i + 1] = acc
#
#     else:
#         # === COLLAPSE EVOLUTION ===
#         for i in range(3):
#             a = a_list[i]
#             v = v_list[i]
#             alpha_i = alpha[i]
#
#             tidal_term = a * aexp * E[i]
#
#             dydt[2 * i] = v
#             coeff = delta0 / 3.0 + 0.5 * (alpha_i - 2.0 / 3.0) * delta0
#             acc = -4.0 * np.pi * rho_u * a * coeff + tidal_term
#             dydt[2 * i + 1] = acc
#
#     return dydt