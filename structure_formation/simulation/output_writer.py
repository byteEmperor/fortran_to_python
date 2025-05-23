# structure_formation/simulation/output_writer.py

def writeHeaders(f1, f2, f3, simulation_params):
    """
    Writes standardized headers and simulation parameters to output files.

    Parameters:
    - f1, f2, f3: File handles for the output files (axes, peculiar velocities, ellipsoid snapshots)
    - simulation_params: SimulationParameters object containing initial conditions

    Returns:
    - header_e: A string representing the header for ellipsoid snapshot data (to be written later)
    """
    # Extract parameters
    Omega0 = simulation_params.Omega0
    e0, e1, e2 = simulation_params.e
    ai = simulation_params.ai
    delta0 = simulation_params.delta
    e11 = simulation_params.e11
    e22 = simulation_params.e22
    e33 = simulation_params.e33

    # --- Header line 1: Parameter labels ---
    param_labels = ["Omega0", "e0", "e1", "e2", "e11", "e22", "e33", "ai", "delta0"]
    label_line = "".join(f"{label:^10}" for label in param_labels) + "\n"
    f1.write(label_line)
    f2.write(label_line)
    f3.write(label_line)

    # --- Header line 2: Parameter values ---
    param_values = [Omega0, e0, e1, e2, e11, e22, e33, ai, delta0]
    value_line = "".join(f"{val:10.5f}" for val in param_values) + "\n"
    f1.write(value_line)
    f2.write(value_line)
    f3.write(value_line)

    # --- Header line 3: Column labels for data ---
    header_a = ["Tau", "exp. factor", "a1", "a2", "a3", "exp. ratio"]
    header_v = ["Tau", "exp. factor", "v1", "v2", "v3", "exp. ratio"]
    header_e = ["exp. factor", "a1", "a2", "a3", "exp. ratio"]

    col_line_a = "".join(f"{col:^15}" for col in header_a) + "\n"
    col_line_v = "".join(f"{col:^15}" for col in header_v) + "\n"
    col_line_e = "".join(f"{col:^15}" for col in header_e) + "\n"

    f1.write(col_line_a)
    f2.write(col_line_v)

    # The ellipsoid column header is typically written later
    return col_line_e
