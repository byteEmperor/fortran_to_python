import numpy as np

def compute_acceleration(a_vec, t, simulation_params):
    # Convert [a1, a2, a3] to full y with zero velocity placeholders
    y_dummy = np.zeros(6)
    y_dummy[0], y_dummy[2], y_dummy[4] = a_vec[0], a_vec[1], a_vec[2]
    y_dummy[1], y_dummy[3], y_dummy[5] = 0.0, 0.0, 0.0  # not used

    # Call your existing derivs
    from structure_formation.models.new_derivs import new_derivs
    dydt = new_derivs(t, y_dummy, simulation_params, None)

    # Extract only accelerations
    acc = np.array([dydt[1], dydt[3], dydt[5]])
    return acc


def leapfrog_integrator(y_start, x1, x2, h, simulation_params, time_params=None):
    num_steps = int((x2 - x1) / h) + 1
    dim = len(y_start)

    # Initialize time and solution arrays
    xp = np.zeros(num_steps)
    yp = np.zeros((num_steps, dim))

    # Split position and velocity
    a = np.array([y_start[0], y_start[2], y_start[4]])  # a1, a2, a3
    v = np.array([y_start[1], y_start[3], y_start[5]])  # v1, v2, v3

    t = x1
    xp[0] = t
    yp[0] = y_start

    for i in range(1, num_steps):
        acc = compute_acceleration(a, t, simulation_params)

        # Kick 1: update velocity to half step
        v_half = v + 0.5 * h * acc

        # Drift: update positions
        a_new = a + h * v_half

        # Kick 2: update velocity full step using new position
        acc_new = compute_acceleration(a_new, t + h, simulation_params)
        v_new = v_half + 0.5 * h * acc_new

        # Store
        t += h
        a = a_new
        v = v_new
        xp[i] = t
        yp[i] = np.array([a[0], v[0], a[1], v[1], a[2], v[2]])

    return xp, yp, None, None
