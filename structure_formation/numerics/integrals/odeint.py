import numpy as np

from structure_formation.simulation.simulation_parameters import SimulationParameters
from structure_formation.models.time_utils import TimeBounds
from structure_formation.models.derivs import derivs
from structure_formation.numerics.integrals.rkqc import rkqc

def odeint(y_start, x1, x2, eps, h1, hmin, simulation_params: SimulationParameters, time_params: TimeBounds, maxstp = 10000, kmax = 500, dxsav = 0.1):
    """
    A port of the Fortran ODEINT with adaptive RK integration using RKQC.

    Parameters:
        y_start : array_like
            Initial values for the ODE system.
        x1, x2 : float
            Start and end times.
        eps : float
            Desired accuracy.
        h1 : float
            Initial step size.
        hmin : float
            Minimum allowed step size.
        derivs : function(t, y)
            Function computing derivatives.
        maxstp : int
            Max number of allowed steps.
        kmax : int
            Max number of points to store in path.
        dxsav : float
            Spacing between saved points.

    Returns:
        xp : list of float
            Time points saved.
        yp : list of np.array
            State vectors at each saved point.
        nok, nbad : int
            Counts of successful and failed steps.
    """

    y = np.array(y_start, dtype=float)
    nvar = len(y)
    x = x1
    h = np.sign(x2 - x1) * abs(h1)
    nok = nbad = 0

    xp = []
    yp = []

    xsav = x - 2 * dxsav
    kount = 0

    for nstp in range(maxstp):
        # Early termination if unphysical values
        if y[0] < 0 or y[2] < 0 or y[4] < 0:
            xp.append(x)
            yp.append(y.copy())
            break

        dydx = derivs(x, y, simulation_params, time_params)
        yscal = np.abs(y) + np.abs(h * dydx) + 1e-30

        # Save if enough time has passed
        if kmax > 0 and abs(x - xsav) > dxsav:
            if kount < (kmax - 1):
                xp.append(x)
                yp.append(y.copy())
                kount += 1
                xsav = x

        # Adjust step if too big for remaining interval
        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x

        y, x_new, hdid, hnext, irkqc = rkqc(y, dydx, x, h, eps, yscal, simulation_params, time_params)
        x = x_new

        if irkqc == 1:
            print("Step too small, exiting integration.")
            break

        if hdid == h:
            nok += 1
        else:
            nbad += 1

        if (x - x2) * (x2 - x1) >= 0.0:
            y_start[:] = y
            if kmax != 0:
                xp.append(x)
                yp.append(y.copy())
            break

        if abs(hnext) < hmin:
            raise RuntimeError("Step size smaller than minimum allowed.")

        h = hnext

    else:
        raise RuntimeError("Too many steps in ODEINT.")

    return xp, yp, nok, nbad
