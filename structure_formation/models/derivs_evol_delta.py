# structure_formation.models.derivs_evol_delta.py

import numpy as np
from structure_formation.numerics.integrals.elliprj import mmlind_scipy

# def derivs_evol_delta(t, y, simulation_params, time_params):
#     """
#     Ellipsoidal evolution with dynamically evolving delta(t)
#     No tides, no Lambda.
#     """
#
#     a1, v1 = y[0], y[1]
#     a2, v2 = y[2], y[3]
#     a3, v3 = y[4], y[5]
#
#     dydt = np.zeros_like(y)
#
#     # Alpha_m from Carlson elliptic projection
#     alpha = [0.0, 0.0, 0.0]
#     alpha[0], _ = mmlind_scipy((a2 / a1)**2, (a3 / a1)**2, 1.0)
#     alpha[1], _ = mmlind_scipy((a1 / a2)**2, (a3 / a2)**2, 1.0)
#     alpha[2], _ = mmlind_scipy((a1 / a3)**2, (a2 / a3)**2, 1.0)
#
#     if t == 0:
#         t = 1e-10
#     rho_u = 1.0 / t**2
#
#     # === Dynamic delta ===
#     # vol_e = max(a1 * a2 * a3, 1e-8)
#     # delta_t = t ** 2 / vol_e - 1
#
#     vol_e = max(a1 * a2 * a3, 1e-8)
#     delta_t = np.clip(t ** 2 / vol_e - 1, -0.99, 100)
#     gamma = 0.01
#
#     for i, (a, v, am) in enumerate(zip([a1, a2, a3], [v1, v2, v3], alpha)):
#         dydt[2 * i] = v
#         coeff = (1 + delta_t) / 3 + 0.5 * (am - 2/3) * delta_t
#         acc = 4 * np.pi * rho_u * coeff * a
#         a_mean = (a1 + a2 + a3) / 3
#         acc += -gamma * (a - a_mean)
#
#         dydt[2 * i + 1] = acc
#
#     return dydt

# def derivs_evol_delta(t, y, simulation_params, time_params):
#     """
#     Stable ellipsoidal void evolution with delta(t) and damped anisotropic feedback.
#     """
#
#     a1, v1 = y[0], y[1]
#     a2, v2 = y[2], y[3]
#     a3, v3 = y[4], y[5]
#
#     dydt = np.zeros_like(y)
#
#     # Elliptic integrals for shape feedback
#     alpha = [0.0, 0.0, 0.0]
#     alpha[0], _ = mmlind_scipy((a2 / a1)**2, (a3 / a1)**2, 1.0)
#     alpha[1], _ = mmlind_scipy((a1 / a2)**2, (a3 / a2)**2, 1.0)
#     alpha[2], _ = mmlind_scipy((a1 / a3)**2, (a2 / a3)**2, 1.0)
#
#     if t == 0:
#         t = 1e-10
#
#     vol_e = max(a1 * a2 * a3, 1e-8)
#     delta_t = t**2 / vol_e - 1.0
#     rho_u = 1.0 / t**2
#
#     # Damp shape feedback over time (like tidal decay)
#     #decay = 1.0 / (1.0 + t)**2
#
#     # Updated (stronger early shaping):
#     decay = 1.0 / (1.0 + t) ** 0.5
#
#     for i, (a, v, am) in enumerate(zip([a1, a2, a3], [v1, v2, v3], alpha)):
#         dydt[2 * i] = v
#
#         # Central expansion term (uniform)
#         base = (1 + delta_t) / 3.0
#
#         # Small shape correction, damped
#         gamma = 1.15
#         correction = gamma * decay * (am - 1.0 / 3.0) * delta_t
#
#         acc = 4.0 * np.pi * rho_u * a * (base + correction)
#         dydt[2 * i + 1] = acc
#
#     return dydt

def derivs_evol_delta(t, y, simulation_params, time_params):
    """
    Stable ellipsoidal void evolution with delta(t), damped anisotropic correction,
    and isotropization force pulling axes toward average size.
    """

    a1, v1 = y[0], y[1]
    a2, v2 = y[2], y[3]
    a3, v3 = y[4], y[5]

    dydt = np.zeros_like(y)

    alpha = [0.0, 0.0, 0.0]
    alpha[0], _ = mmlind_scipy((a2 / a1)**2, (a3 / a1)**2, 1.0)
    alpha[1], _ = mmlind_scipy((a1 / a2)**2, (a3 / a2)**2, 1.0)
    alpha[2], _ = mmlind_scipy((a1 / a3)**2, (a2 / a3)**2, 1.0)

    if t == 0:
        t = 1e-10

    vol_e = max(a1 * a2 * a3, 1e-8)
    delta_t = t**2 / vol_e - 1.0
    rho_u = 1.0 / t**2

    decay = 1.0 / (1.0 + t)**0.5
    gamma = 1.3   # shape correction
    beta = 0.33    # isotropization strength

    a_list = [a1, a2, a3]
    v_list = [v1, v2, v3]
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

    return dydt