# structure_formation/models/integration_initial_conditions.py

from dataclasses import dataclass
import numpy as np

from structure_formation.simulation.simulation_parameters import SimulationParameters

from structure_formation.numerics.integrals.mmlind import mmlind
from .zeropar_functions import openU, closedU

def initialize_conditions(simulation_params: SimulationParameters):
    # Unpack inputs
    e = simulation_params.e
    delta = simulation_params.delta
    Omega0 = simulation_params.Omega0
    ai = simulation_params.ai
    zi = simulation_params.zi
    factor = simulation_params.factor
    aEnd = simulation_params.aEnd
    e11, e22, e33 = simulation_params.tides.e11, simulation_params.tides.e22, simulation_params.tides.e33

    # Legacy constants from fortran
    one = 1.0
    two = 2.0
    half = 0.5
    twothr = 2.0 / 3.0

    # Initial axis values
    a1i, a2i, a3i = e

    # Hubble parameter at initial time
    Hi = np.sqrt(4.0 / 3.0) * np.sqrt((1 + Omega0 * zi) / (Omega0 + Omega0 * zi))

    # Omega at initial time
    Omega_init = Omega0 * (1 + zi) / (1 + Omega0 * zi)

    # Compute alpha terms via mmlind
    b1, b2, b3 = e
    b12, b22, b32 = b1 ** 2, b2 ** 2, b3 ** 2

    alpha = [0.0, 0.0, 0.0]
    alpha[0], ierr = mmlind(b32, b22, b12)
    alpha[1], ierr = mmlind(b12, b32, b22)
    alpha[2], ierr = mmlind(b12, b22, b32)

    alpha = [twothr * b1 * b2 * b3 * a for a in alpha]

    # Peculiar velocities
    vpec = [
        -half * Hi * Omega_init ** 0.6 * alpha[0] * (delta - one) - half * Hi * Omega_init ** 0.6 * e11 * ai,
        -half * Hi * Omega_init ** 0.6 * alpha[1] * (delta - one) - half * Hi * Omega_init ** 0.6 * e22 * ai,
        -half * Hi * Omega_init ** 0.6 * alpha[2] * (delta - one) - half * Hi * Omega_init ** 0.6 * e33 * ai,
    ]

    # Full initial velocities
    v1i = Hi * a1i + vpec[0] * factor
    v2i = Hi * a2i + vpec[1] * factor
    v3i = Hi * a3i + vpec[2] * factor

    # Initial state vector
    y0 = np.array([a1i, v1i, a2i, v2i, a3i, v3i])

    return y0, Hi, Omega_init, alpha, vpec
