import numpy as np

def tester(t, y, simulation_params, time_params):
    """
    Derivative function for ellipsoidal void evolution in an EdS universe.

    Features:
    - For delta ≈ 0: each axis expands as a(t) ∝ t^{2/3}, with d²a/dt² ∝ t^{-4/3}
    - For delta < 0: expansion with shape-restoring force toward sphericity
    - Velocity remains positive; no axis crossover
    - Structure mimics realistic void evolution from theory

    Parameters:
    - t : float
        Physical time.
    - y : array-like (6,)
        State vector: [a1, v1, a2, v2, a3, v3]
    - simulation_params : object with .delta
    - time_params : unused (placeholder)

    Returns:
    - dydt : array-like (6,)
        Derivatives of state vector
    """

    delta = simulation_params.delta

    # Unpack axes and velocities
    a1, v1 = y[0], y[1]
    a2, v2 = y[2], y[3]
    a3, v3 = y[4], y[5]
    axes = np.array([a1, a2, a3])
    velocities = np.array([v1, v2, v3])
    dydt = np.zeros_like(y)

    if t <= 0:
        t = 1e-10  # avoid singularity

    # === Homogeneous case: pure EdS expansion ===
    if abs(delta) < 1e-8:
        for i in range(3):
            dydt[2*i] = velocities[i]
            dydt[2*i + 1] = -2 / (9 * t**(4/3)) * axes[i]
        return dydt

    # === Underdense case: apply expansion + shape-restoring correction ===
    mean_a = np.mean(axes)

    for i in range(3):
        a = axes[i]
        v = velocities[i]

        # 1. EdS gravitational-like expansion
        grav_acc = -2 / (9 * t**(4/3))

        # 2. Shape-restoring term (scaled weaker early on)
        restoring_force = -0.15 * (a - mean_a) / max(t**0.5, 1e-5)

        # 3. Total acceleration
        acc = grav_acc + restoring_force

        # 4. Clamp: no negative velocities or accelerations
        dydt[2*i + 1] = max(acc, 0.0)
        dydt[2*i] = max(v, 0.0)

    return dydt