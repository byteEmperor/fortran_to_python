# structure_formation/tests/test_simulation_behaviour.py

import numpy as np

import pytest
from structure_formation.simulation.simulation_parameters import SimulationParameters
from structure_formation.simulation.scipy_style_sim import run_integration_scipy

# === 1. Symmetry ===

def check_symmetry(output, tol=1e-6):
    final = output[-1]['axes']
    symmetric = all(abs(final[i] - final[j]) < tol for i in range(3) for j in range(i + 1, 3))
    return symmetric, {"final_axes": final}


# === 2. Expansion (underdensity sanity check) ===

def check_expanding(output):
    a_start = output[0]['axes'][0]
    a_end = output[-1]['axes'][0]
    return a_end > a_start, {"a_start": a_start, "a_end": a_end}


# === 3. Collapse (overdensity sanity check) ===

def check_collapsing(output):
    a_start = output[0]['axes'][0]
    a_end = output[-1]['axes'][0]
    return a_end < a_start, {"a_start": a_start, "a_end": a_end}


# === 4. Underdensity: sphericity improves ===

def check_underdensity_behavior(output):
    errors = []

    for row in output:
        a1, a2, a3 = row['axes']
        r12 = max(a1 / a2, a2 / a1)
        r13 = max(a1 / a3, a3 / a1)
        r23 = max(a2 / a3, a3 / a2)
        error = max(abs(r12 - 1), abs(r13 - 1), abs(r23 - 1))
        errors.append(error)

    start_error = errors[0]
    end_error = errors[-1]
    passed = end_error < start_error

    return passed, {
        "start_error": start_error,
        "end_error": end_error,
        "errors": errors[:10] + ['...'] + errors[-5:],
        "start_axes": output[0]['axes'],
        "end_axes": output[-1]['axes'],
    }


# === Parametrized Test Cases ===

@pytest.mark.parametrize(
    "e, delta, check_fn, description",
    [
        ([1.0, 1.0, 1.0], 0.0, check_symmetry, "Symmetry should be preserved for spherical delta=0"),
        ([1.0, 0.8, 0.6], -0.5, check_underdensity_behavior, "Underdensity should expand and approach sphericity"),
        ([1.0, 0.8, 0.6], 0.5, check_collapsing, "Overdensity should collapse ellipsoid"),
    ]
)
def test_simulation_behavior(e, delta, check_fn, description):
    sim_params = SimulationParameters(
        Omega0=1.0,
        e=e,
        ai=0.1,
        zi=1.0 / 0.1 - 1.0,
        delta=delta,
        aEnd=1.0,
        e11=0.01,
        e22=0.0,
        e33=-0.1
    )

    output = run_integration_scipy(sim_params)
    passed, data = check_fn(output)

    print(f"\n[TEST] {description}")
    for key, val in data.items():
        print(f"  {key}: {val}")

    assert passed is True, (
        f"[FAIL] {description}\nDetails:\n" +
        "\n".join(f"  {k}: {v}" for k, v in data.items())
    )