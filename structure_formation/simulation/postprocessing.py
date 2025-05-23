# structure_formation/simulation/postprocessing.py

import math
import numpy as np
from structure_formation.models.zeropar_functions import openU, closedU
from structure_formation.numerics.roots.zbrent import zbrent
from structure_formation.simulation.simulation_parameters import SimulationParameters

def postprocessODE(output, f1, f2, f3, header_e: str, simulation_params: SimulationParameters):
    Omega0 = simulation_params.Omega0
    zi = simulation_params.zi
    ai = simulation_params.ai

    # Precompute expansion bounds
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

    for row in output:
        tau = row["tau"]
        aexp = row["aexp"]
        a1, a2, a3 = row["axes"]
        vpec1, vpec2, vpec3 = row["peculiar_velocities"]

        # Write scale factor output
        f1.write(f"{tau:14.6f}{aexp:14.6f}{a1:14.6f}{a2:14.6f}{a3:14.6f}{(aexp / ai):14.6f}\n")

        # Write ellipsoid snapshot if applicable
        if (aexp > aexpwr) or (row == output[-1]):
            nwr += 1
            texpwr = float(nwr) * texpwrdl
            aexpwr = texpwr ** (1.0 / 1.5)
            scratch_lines.append(f"{aexp:14.6f}{a1:14.6f}{a2:14.6f}{a3:14.6f}{(aexp / ai):14.6f}\n")

        # Write peculiar velocities (+1 offset preserved)
        vpec = np.clip([vpec1, vpec2, vpec3], -1e4, 1e4)
        f2.write(f"{tau:14.6f}{aexp:14.6f}{vpec[0]+1:14.6f}{vpec[1]+1:14.6f}{vpec[2]+1:14.6f}{(aexp / ai):14.6f}\n")

    # Final ellipsoid output
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
