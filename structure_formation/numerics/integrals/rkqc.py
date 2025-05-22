import numpy as np

from structure_formation.numerics.integrals.rk4 import rk4
from structure_formation.simulation.simulation_parameters import SimulationParameters
from structure_formation.models.time_utils import TimeBounds
from structure_formation.models.derivs import derivs

def rkqc(y, dydx, x, htry, eps, yscal, simulation_params: SimulationParameters, time_params: TimeBounds):
    SAFETY = 0.9
    ERRCON = 6.0e-4
    PGROW = -0.2
    PSHRNK = -0.25
    FCOR = 1.0 / 15.0  # ≈ 0.0666667
    NMAX = len(y)

    h = htry
    xsav = x
    ysav = y.copy()
    dysav = dydx.copy()

    while True:
        hh = 0.5 * h
        ytemp = rk4(ysav, dysav, xsav, hh, simulation_params, time_params)
        xmid = xsav + hh
        dym = derivs(xmid, ytemp, simulation_params, time_params)
        yfull = rk4(ytemp, dym, xmid, hh, simulation_params, time_params)

        xnew = xsav + h
        ybig = rk4(ysav, dysav, xsav, h, simulation_params, time_params)

        errmax = np.max(np.abs((yfull - ybig) / yscal)) / eps

        if errmax <= 1.0:
            hdid = h
            if errmax > ERRCON:
                hnext = SAFETY * h * errmax**PGROW
            else:
                hnext = 4.0 * h
            y = yfull + (yfull - ybig) * FCOR
            return y, xnew, hdid, hnext, 0
        else:
            h = SAFETY * h * errmax**PSHRNK
            if xnew == xsav:
                return y, x, h, h, 1  # Failure to progress
