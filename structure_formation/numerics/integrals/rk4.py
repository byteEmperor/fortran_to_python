import numpy as np

from structure_formation.models.derivs import derivs
from structure_formation.simulation.simulation_parameters import SimulationParameters
from structure_formation.models.time_utils import TimeBounds

def rk4(y, dydx, x, h, simulation_params: SimulationParameters, time_params: TimeBounds):
    n = len(y)
    hh = 0.5 * h
    h6 = h / 6.0
    xh = x + hh

    yt = y + hh * dydx
    dyt = derivs(xh, yt, simulation_params, time_params)

    yt = y + hh * dyt
    dym = derivs(xh, yt, simulation_params, time_params)

    yt = y + h * dym
    dym = dyt + dym

    dyt = derivs(x + h, yt, simulation_params, time_params)
    yout = y + h6 * (dydx + dyt + 2.0 * dym)

    return yout
