import numpy as np

def chat_derivs(t, y, simulation_params, time_params):
    """
    Derivs function for ellipsoidal void evolution with:
    - Time-evolving delta(t)
    - a(t) ∝ t^(2/3), d²a/dt² ∝ t^(-4/3) when delta ~ 0
    - Axis ratios tend to 1 for underdensities
    - Velocities stay positive

    Inputs:
    - t: time (float)
    - y: state vector: [a1, v1, a2, v2, a3, v3]
    - simulation_params: has no longer static delta
    - time_params: (unused)

    Returns:
    - dydt: derivatives
    """

    # Unpack state vector
    a1, v1 = y[0], y[1]
    a2, v2 = y[2], y[3]
    a3, v3 = y[4], y[5]
    dydt = np.zeros_like(y)

    # Avoid division by zero
    if t == 0:
        t = 1e-10

    # === Compute time-dependent overdensity
    volume = np.clip(a1 * a2 * a3, 1e-8, 1e8)
    delta_t = t**2 / volume - 1.0

    # === Velocity updates (positions)
    dydt[0] = v1
    dydt[2] = v2
    dydt[4] = v3

    # === EdS-like behavior if delta_t ≈ 0
    if abs(delta_t) < 1e-6:
        for i, a in enumerate([a1, a2, a3]):
            dydt[2*i + 1] = -2 / (9 * t**(4/3)) * a
        return dydt

    # === Underdensity: promote isotropic expansion
    axes = np.array([a1, a2, a3])
    velocities = np.array([v1, v2, v3])
    mean_a = np.mean(axes)

    for i in range(3):
        a = np.clip(axes[i], 1e-8, 1e5)
        v = velocities[i]

        # Shape restoring force
        shape_eq = -0.5 * delta_t * (a - mean_a) / max(t, 1e-5)

        # Total matter acceleration (gravity + shape feedback)
        matter_acc = -2 / (9 * t**(4/3)) + shape_eq
        matter_acc = np.clip(matter_acc, -100, 100)

        dydt[2*i + 1] = max(matter_acc * a, 0.0)
        dydt[2*i] = max(v, 0.0)

    return dydt
