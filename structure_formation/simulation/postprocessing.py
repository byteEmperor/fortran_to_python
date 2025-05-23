# structure_formation/simulation/postprocessing.py

import math
import numpy as np
from structure_formation.models.zeropar_functions import openU, closedU
from structure_formation.numerics.roots.zbrent import zbrent
from structure_formation.simulation.simulation_parameters import SimulationParameters

def postprocessODE(sol, f1, f2, f3, header_e: str, simulation_params: SimulationParameters):
    Omega0 = simulation_params.Omega0
    zi = simulation_params.zi
    ai = simulation_params.ai

    aexpwrmx = 1.0
    aexpwrmn = 0.0
    texpwrmx = aexpwrmx ** 1.5
    texpwrmn = aexpwrmn ** 1.5
    texpwrrn = texpwrmx - texpwrmn

    scratch_lines = []

    nmwr = 1000
    texpwrdl = texpwrrn / float(nmwr - 1) if nmwr > 1 else 0.0
    nwr = 0
    aexpwr = texpwrmn ** (1.0 / 1.5)

    for i in range(len(sol.t)):
        tau = sol.t[i]

        # Compute aexp from conformal time
        if abs(Omega0 - 1.0) < 1e-8:
            aexp = (3.0 ** (1.0 / 3.0)) * (tau ** (2.0 / 3.0)) / (1.0 + zi)
        elif Omega0 < 1.0:
            psi = zbrent(lambda x: openU(x, sigma=compute_sigma(Omega0, zi), ztau=tau), 0.0, 20.0, 1e-6)
            aexp = (math.cosh(psi) - 1.0) * Omega0 / (2.0 * (1.0 - Omega0))
        else:
            theta = zbrent(lambda x: closedU(x, sigma=compute_sigma(Omega0, zi), ztau=tau), -5.0, 5.0, 1e-6)
            aexp = (1.0 - math.cos(theta)) * Omega0 / (2.0 * (Omega0 - 1.0))

        z = 1.0 / aexp - 1.0
        Hz = (1.0 + z) * math.sqrt(1.0 + Omega0 * z) / ((1.0 + zi) ** 1.5) / math.sqrt(Omega0)
        Hz *= math.sqrt(4.0 / 3.0)

        # Write a_i output
        f1.write(f"{tau:14.6f}{aexp:14.6f}{sol.y[0, i]:14.6f}{sol.y[2, i]:14.6f}{sol.y[4, i]:14.6f}{(aexp / ai):14.6f}\n")

        # Write ellipsoid snapshot if needed
        if (aexp > aexpwr) or (i == len(sol.t) - 1):
            nwr += 1
            texpwr = float(nwr) * texpwrdl
            aexpwr = texpwr ** (1.0 / 1.5)
            scratch_lines.append(f"{aexp:14.6f}{sol.y[0, i]:14.6f}{sol.y[2, i]:14.6f}{sol.y[4, i]:14.6f}{(aexp / ai):14.6f}\n")

        # Peculiar velocity calculations with clamping
        vpec = [(sol.y[1, i] - Hz * sol.y[0, i]) / (Hz * sol.y[0, i]),
                (sol.y[3, i] - Hz * sol.y[2, i]) / (Hz * sol.y[2, i]),
                (sol.y[5, i] - Hz * sol.y[4, i]) / (Hz * sol.y[4, i])]
        vpec = np.clip(vpec, -1e4, 1e4)

        f2.write(f"{tau:14.6f}{aexp:14.6f}{(vpec[0]+1):14.6f}{(vpec[1]+1):14.6f}{(vpec[2]+1):14.6f}{(aexp / ai):14.6f}\n")

    # Write ellipsoid output
    f3.write(f"Number of \"drawing\" timesteps: {nwr:6d}\n")
    f3.write(header_e)
    for line in scratch_lines:
        f3.write(line)

    print("Calculation complete. Output files written.")

def compute_sigma(Omega0: float, zi: float) -> float:
    sqrt_3_4 = math.sqrt(3.0 / 4.0)
    if Omega0 > 1.0:
        return sqrt_3_4 * (Omega0 / (Omega0 - 1)) ** 1.5 / 2 * (1 + zi) ** 1.5
    elif Omega0 < 1.0:
        return sqrt_3_4 * (Omega0 / (1 - Omega0)) ** 1.5 / 2 * (1 + zi) ** 1.5
    else:
        return 0.0
