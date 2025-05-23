import numpy as np
from scipy.optimize import brentq

from structure_formation.simulation.simulation_parameters import SimulationParameters
from structure_formation.models.time_utils import TimeBounds, compute_conformal_time_bounds
from structure_formation.models.integration_initial_conditions import initialize_conditions
from structure_formation.numerics.integrals.leapfrog_integrator import leapfrog_integrator
from structure_formation.models.zeropar_functions import openU, closedU


def run_integration_leapfrog(simulation_params: SimulationParameters):
    time_params: TimeBounds = compute_conformal_time_bounds(simulation_params)
    y0, Hi, Omega_init, alpha, vpec = initialize_conditions(simulation_params)

    h = time_params.delta_tau / 499.0

    xp, yp, _, _ = leapfrog_integrator(
        y_start=y0,
        x1=time_params.tau_init,
        x2=time_params.tau_end,
        h=h,
        simulation_params=simulation_params,
        time_params=time_params
    )

    output = []

    for i in range(len(xp)):
        tau = xp[i]

        # Compute aexp from tau
        if abs(simulation_params.Omega0 - 1.0) < 1e-8:
            aexp = (3.0 ** (1.0 / 3.0)) * (tau ** (2.0 / 3.0)) / (1.0 + simulation_params.zi)
        elif simulation_params.Omega0 < 1.0:
            psi = brentq(lambda x: openU(x, tau, simulation_params.Omega0), 0.0, 20.0, xtol=1e-6)
            aexp = (np.cosh(psi) - 1.0) * simulation_params.Omega0 / (2.0 * (1.0 - simulation_params.Omega0))
        else:
            theta = brentq(lambda x: closedU(x, tau, simulation_params.Omega0), -5.0, 5.0, xtol=1e-6)
            aexp = (1.0 - np.cos(theta)) * simulation_params.Omega0 / (2.0 * (simulation_params.Omega0 - 1.0))

        z = 1.0 / aexp - 1.0
        Hz = (1 + z) * np.sqrt(1 + simulation_params.Omega0 * z) / ((1 + simulation_params.zi) ** 1.5) / np.sqrt(simulation_params.Omega0)
        Hz *= np.sqrt(4.0 / 3.0)

        a1, v1 = yp[i][0], yp[i][1]
        a2, v2 = yp[i][2], yp[i][3]
        a3, v3 = yp[i][4], yp[i][5]

        vpec1 = (v1 - Hz * a1) / (Hz * a1)
        vpec2 = (v2 - Hz * a2) / (Hz * a2)
        vpec3 = (v3 - Hz * a3) / (Hz * a3)

        vpec1 = max(min(vpec1, 1e4), -1e4)
        vpec2 = max(min(vpec2, 1e4), -1e4)
        vpec3 = max(min(vpec3, 1e4), -1e4)

        output.append({
            'tau': tau,
            'aexp': aexp,
            'z': z,
            'axes': (a1, a2, a3),
            'velocities': (v1, v2, v3),
            'peculiar_velocities': (vpec1, vpec2, vpec3),
            'Hz': Hz,
            'aexp/ai': aexp / simulation_params.ai,
        })

    print(f"Leapfrog integration done: kount={len(xp)}")

    return output
