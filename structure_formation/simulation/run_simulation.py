# structure_formation/simulation/run_simulation.py

from structure_formation.simulation.fortran_style_sim import run_integration_fortran
from structure_formation.simulation.simulation_parameters import SimulationParameters
from structure_formation.simulation.output_config import OutputConfig
from structure_formation.simulation.scipy_style_sim import run_integration_scipy
from structure_formation.simulation.scipy_style_sim2 import run_integration_scipy2
from structure_formation.simulation.output_writer import writeHeaders
from structure_formation.simulation.postprocessing import postprocessODE

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


def main():

    sim1: SimulationParameters = create_simulation_parameters(
        Omega0=1.0,
        axes=[1.0, 0.8, 0.6],
        ai=0.1,
        delta=0,
        aEnd=1.0
    )

    output1: OutputConfig = OutputConfig(
        ai_path="a_1.txt",
        vpec_path="v_1.txt",
        ellipsoid_path="e_1.txt"
    )

    write(output1, sim1, run_integration_fortran(sim1))


if __name__ == "__main__":
    main()
