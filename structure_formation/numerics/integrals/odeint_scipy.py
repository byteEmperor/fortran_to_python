# structure_formation/numerics/integrals/odeint_scipy.py

from structure_formation.models.derivs import derivs
from structure_formation.models.new_derivs import new_derivs
from structure_formation.models.derivs_evol_delta import derivs_evol_delta

import numpy as np
from scipy.integrate import solve_ivp

def odeint_scipy(y_start, x1, x2, eps, h1, hmin, simulation_params, time_params, maxstp=10000, kmax=500, dxsav=0.1):
    y0 = np.array(y_start, dtype=float)
    nvar = len(y0)

    # Create evaluation points for output, spaced by dxsav, capped by kmax
    if kmax > 0:
        num_points = kmax
        t_eval = np.linspace(x1, x2, num_points)
    else:
        t_eval = None  # Let solve_ivp choose points

    # Define event to stop integration if any of y[0], y[2], y[4] < 0
    def stop_if_negative(t, y):
        # Minimum of y[0], y[2], y[4], stop when negative or zero
        return min(y[0], y[2], y[4])  # zero crossing triggers event

    stop_if_negative.terminal = True
    stop_if_negative.direction = -1  # only trigger when crossing from positive to negative

    # Wrapper derivative function for solve_ivp
    def fun(t, y):
        return derivs_evol_delta(t, y, simulation_params, time_params)

    sol = solve_ivp(
        fun=fun,
        t_span=(x1, x2),
        y0=y0,
        method='RK45',
        t_eval=t_eval,
        rtol=eps,
        atol=eps*0.1,
        events=stop_if_negative,
        max_step=h1  # starting guess, but max step is enforced here
    )

    # After integration, prepare outputs matching old format
    xp = list(sol.t)
    yp = [sol.y[:, i].copy() for i in range(len(sol.t))]  # list of state arrays per time step

    # Solve_ivp doesn't expose nok, nbad. Approximate as:
    nok = None  # number of accepted steps is internal; no direct access
    nbad = None

    if sol.status == 1:
        print("Integration stopped early due to negative value event.")

    if sol.status < 0:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return xp, yp, nok, nbad
