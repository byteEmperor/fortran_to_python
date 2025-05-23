# structure_formation/simulation/scipy_style_sim2.py

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from structure_formation.simulation.simulation_parameters import SimulationParameters
from structure_formation.models.time_utils import TimeBounds, compute_conformal_time_bounds
from structure_formation.models.integration_initial_conditions import initialize_conditions
from structure_formation.numerics.integrals.odeint_scipy import odeint_scipy
from structure_formation.simulation.postprocessing import postprocessODE
from structure_formation.models.zeropar_functions import openU, closedU

def run_integration_scipy2(simulation_params: SimulationParameters, output_config):
    # Step 1: Prepare simulation
    time_params: TimeBounds = compute_conformal_time_bounds(simulation_params)
    y0, Hi, Omega_init, alpha, vpec = initialize_conditions(simulation_params)

    # Step 2: Integrator settings
    eps = 1e-6
    h1 = time_params.delta_tau / (simulation_params.aEnd * 100000.0)
    hmin = 0.0
    kmax = 499
    dxsav = time_params.delta_tau / kmax

    # Step 3: Run integration
    xp, yp, nok, nbad = odeint_scipy(
        y_start=y0,
        x1=time_params.tau_init,
        x2=time_params.tau_end,
        eps=eps,
        h1=h1,
        hmin=hmin,
        simulation_params=simulation_params,
        time_params=time_params,
        maxstp=None,
        kmax=kmax,
        dxsav=dxsav
    )

    print(f"Integration complete: kount={len(xp)}, nok={nok}, nbad={nbad}")

    # Step 4: Package results into `sol`-like object to mimic solve_ivp output
    class Solution:
        def __init__(self, t, y):
            self.t = np.array(t)
            self.y = np.array(y).T  # shape (nvars, npoints)

    sol = Solution(xp, yp)

    # Step 5: Open output files
    with open(output_config.ai_path, "w") as f1, \
         open(output_config.vpec_path, "w") as f2, \
         open(output_config.ellipsoid_path, "w") as f3:
        postprocessODE(sol, f1, f2, f3, header_e="aexp a1 a2 a3 aexp/ai\n", simulation_params=simulation_params)

    return 0
