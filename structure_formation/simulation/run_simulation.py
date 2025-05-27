# structure_formation/simulation/run_simulation.py

import os
import numpy as np

from structure_formation.simulation.fortran_style_sim import run_integration_fortran
from structure_formation.simulation.simulation_parameters import SimulationParameters
from structure_formation.simulation.output_config import OutputConfig
from structure_formation.simulation.scipy_style_sim import run_integration_scipy
from structure_formation.simulation.output_writer import writeHeaders
from structure_formation.simulation.postprocessing import postprocessODE
from structure_formation.simulation.leapfrog_style_sim import run_integration_leapfrog

def write(output_config: OutputConfig, simulation_parameters: SimulationParameters, output):
    with open(output_config.ai_path, "w") as f1, \
        open(output_config.vpec_path, "w") as f2, \
        open(output_config.ellipsoid_path, "w") as f3:

        header_e = writeHeaders(f1, f2, f3, simulation_parameters)

        postprocessODE(output, f1, f2, f3, header_e, simulation_parameters)


def create_simulation_parameters(Omega0, axes, ai, delta, aEnd):
    simulation_parameters = SimulationParameters(
        Omega0=Omega0,
        e = axes,
        ai = ai,
        zi = 1.0 / ai - 1.0,
        delta = delta,
        aEnd = aEnd
    )

    return simulation_parameters

def delta_sweep(base_config_dir, delta_values, base_sim_params: SimulationParameters, simulation_function):
    os.makedirs(base_config_dir, exist_ok=True)

    # takes either run_integration_fortran or run_integration_scipy

    for i, delta in enumerate(delta_values):
        # Clone base parameters and change delta
        sim_params = SimulationParameters(
            Omega0=base_sim_params.Omega0,
            e=base_sim_params.e,
            ai=base_sim_params.ai,
            zi=base_sim_params.zi,
            delta=delta,
            aEnd=base_sim_params.aEnd,
            e11=base_sim_params.e11,
            e22=base_sim_params.e22,
            e33=base_sim_params.e33
        )

        # Create unique file names per delta
        suffix = f"d{i:03d}"  # or f"d{delta:+.2f}".replace('.', 'p').replace('-', 'm') for delta in filename

        output_config = OutputConfig(
            ai_path=os.path.join(base_config_dir, f"ai_{suffix}.txt"),
            vpec_path=os.path.join(base_config_dir, f"vpec_{suffix}.txt"),
            ellipsoid_path=os.path.join(base_config_dir, f"ellipsoid_{suffix}.txt"),
            logs_path=os.path.join(base_config_dir, f"log_{suffix}.txt"),
        )

        print(f"[Run {i}] Running simulation with delta = {delta}")
        output = simulation_function(sim_params)
        write(output_config, sim_params, output)


def main():

    sim1: SimulationParameters = create_simulation_parameters(
        Omega0=1.0,
        axes=[1.0, 0.8, 0.6],
        ai=0.1,
        delta=0.0,
        aEnd=1.0
    )

    output1: OutputConfig = OutputConfig(
        ai_path="a_1.txt",
        vpec_path="v_1.txt",
        ellipsoid_path="e_1.txt"
    )

    #write(output1, sim1, run_integration_fortran(sim1))
    delta_vals = np.linspace(-0.5, -0.5, 1)
    delta_sweep("temp", delta_vals, sim1, run_integration_leapfrog)
    #write(output1, sim1, run_integration_fortran(sim1))
    #delta_sweep("temp2", delta_vals, sim1, run_integration_scipy)


if __name__ == "__main__":
    main()
