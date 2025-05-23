# structure_formation/models/time_utils.py

import math
from dataclasses import dataclass
from structure_formation.simulation.simulation_parameters import SimulationParameters

@dataclass
class TimeBounds:
    tau_init: float
    tau_end: float
    delta_tau: float
    sigma: float

def compute_conformal_time_bounds(simulation_params: SimulationParameters, a0: float = 1.0) -> TimeBounds:

    """

    This function sets up the cosmic time parameterization (converting scale factor 'a' or redshift 'z'
    into the corresponding conformal time tau -- essential for evolving the universe forward in time
    depending on Omega0

    """

    e = simulation_params.e
    delta = simulation_params.delta
    Omega0 = simulation_params.Omega0
    ai = simulation_params.ai
    zi = simulation_params.zi
    factor = simulation_params.factor
    aEnd = simulation_params.aEnd
    e11, e22, e33 = simulation_params.e11, simulation_params.e22, simulation_params.e33

    one = 1.0
    two = 2.0
    half = 0.5
    sqrt_3 = math.sqrt(3.0)
    sqrt_3_4 = math.sqrt(3.0 / 4.0)
    sigma = 0.0

    if abs(Omega0 - one) < 1e-8:
        tau_init = sqrt_3 / 3.0
        tau_end = (aEnd ** 1.5) * ((1 + zi) ** 1.5) * sqrt_3 / 3.0
    elif Omega0 > one:
        help_val = 1 - ((ai / a0) * (2 * Omega0 - 2) / Omega0)
        thet = math.acos(help_val)
        tau_init = ((thet - math.sin(thet)) * (Omega0 / (Omega0 - 1)) ** 1.5) / 2
        tau_init *= ((1 + zi) ** 1.5) * sqrt_3_4

        sigma = sqrt_3_4 * (Omega0 / (Omega0 - 1)) ** 1.5 / 2
        sigma *= (1 + zi) ** 1.5

        help_val = 1 - ((aEnd / a0) * (2 * Omega0 - 2) / Omega0)
        thet = math.acos(help_val)
        tau_end = ((thet - math.sin(thet)) * (Omega0 / (Omega0 - 1)) ** 1.5) / 2
        tau_end *= ((1 + zi) ** 1.5) * sqrt_3_4

    elif Omega0 < one:
        help_val = 1 + (ai / a0) * (2 - 2 * Omega0) / Omega0
        psi = math.log(help_val + math.sqrt(help_val**2 - 1))
        tau_init = (math.sinh(psi) - psi) * (Omega0 / (1 - Omega0)) ** 1.5
        tau_init *= ((1 + zi) ** 1.5) * sqrt_3_4 / 2

        sigma = sqrt_3_4 * (Omega0 / (1 - Omega0)) ** 1.5
        sigma *= ((1 + zi) ** 1.5) / 2

        help_val = 1 + (aEnd / a0) * (2 - 2 * Omega0) / Omega0
        psi = math.log(help_val + math.sqrt(help_val**2 - 1))
        tau_end = (math.sinh(psi) - psi) * (Omega0 / (1 - Omega0)) ** 1.5
        tau_end *= ((1 + zi) ** 1.5) * sqrt_3_4 / 2

    else:
        raise ValueError("Invalid value for Omega0")

    delta_tau = tau_end - tau_init

    print(f"tau_init: {tau_init}")
    print(f"tau_end:  {tau_end}")
    print(f"sigma:    {sigma}")

    return TimeBounds(tau_init=tau_init, tau_end=tau_end, delta_tau=delta_tau, sigma=sigma)
