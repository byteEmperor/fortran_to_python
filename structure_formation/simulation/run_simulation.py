# structure_formation/simulation/run_simulation.py

from structure_formation.simulation.fortran_style_sim import run_integration
from structure_formation.simulation.simulation_parameters import SimulationParameters
from structure_formation.simulation.output_config import OutputConfig
from structure_formation.simulation.scipy_style_sim import run_integration_scipy

def fortran_style(output_config: OutputConfig, simulation_parameters: SimulationParameters):

    output = run_integration(simulation_parameters)

    # "a_i's" → xp(i), aexp, a1, a2, a3, aexp/ai
    with open(output_config.ai_path, "w") as f:
        for row in output:
            f.write(f"{row['tau']:14.6f} {row['aexp']:14.6f} {row['axes'][0]:14.6f} "
                    f"{row['axes'][1]:14.6f} {row['axes'][2]:14.6f} {row['aexp/ai']:14.6f}\n")

    return 0

def scipy_style(output_config: OutputConfig, simulation_parameters: SimulationParameters):

    output = run_integration_scipy(simulation_parameters)

    with open(output_config.ai_path, "w") as f:
        for row in output:
            f.write(f"{row['tau']:14.6f} {row['aexp']:14.6f} {row['axes'][0]:14.6f} "
                    f"{row['axes'][1]:14.6f} {row['axes'][2]:14.6f} {row['aexp/ai']:14.6f}\n")


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

    simulation_parameters: SimulationParameters = create_simulation_parameters(1.0, [1.0, 0.8, 0.6],
                                                                               0.1, 0.2, 1.0)

    sim2: SimulationParameters = create_simulation_parameters(1.0, [1.0, 0.8, 0.6], 0.1, -0.5, 1)
    sim3: SimulationParameters = create_simulation_parameters(1.0, [0.8, 0.8, 0.8], 0.1, 5, 10)

    output_config: OutputConfig = OutputConfig(
        ai_path="a2.txt",
        vpec_path="v.txt",
        ellipsoid_path="e.txt"
    )
    output_config2: OutputConfig = OutputConfig(
        ai_path="a3.txt",
        vpec_path="v.txt",
        ellipsoid_path="e.txt"
    )

    #fortran_style(output_config, simulation_parameters)
    scipy_style(output_config, sim2)
    scipy_style(output_config2, sim3)


if __name__ == "__main__":
    main()
