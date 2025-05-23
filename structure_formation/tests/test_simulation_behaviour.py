# structure_formation/tests/test_simulation_behaviour.py

import numpy as np

import pytest
from structure_formation.simulation.simulation_parameters import SimulationParameters
from structure_formation.simulation.scipy_style_sim import run_integration_scipy

# ------------------------
# Test helpers
# ------------------------

def check_symmetry(output, tol=1e-6):
    final = output[-1]['axes']
    symmetric = all(abs(final[i] - final[j]) < tol for i in range(3) for j in range(i + 1, 3))
    return symmetric, final

def check_expanding(output):
    a_start = output[0]['axes'][0]
    a_end = output[-1]['axes'][0]
    return a_end > a_start, (a_start, a_end)

def check_collapsing(output):
    a_start = output[0]['axes'][0]
    a_end = output[-1]['axes'][0]
    return a_end < a_start, (a_start, a_end)

def check_underdensity_behavior(output, tol=0.05):
    a_start = output[0]['axes']
    a_end = output[-1]['axes']

    # Check all axes expanded
    expanded = all(a_end[i] > a_start[i] for i in range(3))

    # Compute sphericity ratio at each time
    ratios = []
    for row in output:
        axes = row['axes']
        max_a = max(axes)
        min_a = min(axes)
        ratios.append(max_a / min_a)

    start_ratio = ratios[0]
    end_ratio = ratios[-1]
    trend_toward_sphericity = all(r <= start_ratio + tol for r in ratios[1:]) and end_ratio < start_ratio

    return expanded and trend_toward_sphericity, {
        "start_axes": a_start,
        "end_axes": a_end,
        "start_ratio": start_ratio,
        "end_ratio": end_ratio,
        "ratios": ratios[:10] + ['...'] + ratios[-5:]
    }




# ------------------------
# Parametrized tests
# ------------------------

@pytest.mark.parametrize(
    "e, delta, check_fn, description",
    [
        ([1.0, 1.0, 1.0], 0.0, check_symmetry, "Symmetry should be preserved for spherical delta=0"),
        ([1.0, 0.8, 0.6], -0.5, check_underdensity_behavior, "Underdensity should expand ellipsoid"),
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
        aEnd=1.0
    )

    output = run_integration_scipy(sim_params)
    passed, data = check_fn(output)

    print(f"\n[TEST] {description}")
    if isinstance(data, dict):
        print(f"  Start Axes: {data['start_axes']}")
        print(f"  End Axes:   {data['end_axes']}")
        print(f"  Sphericity: start={data['start_ratio']:.4f}, end={data['end_ratio']:.4f}")
        if isinstance(data['ratios'], list):
            print(f"  Ratio samples: {data['ratios']}")





