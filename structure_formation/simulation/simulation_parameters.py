# structure_formation/simulation/simulation_parameters.py

from dataclasses import dataclass
from structure_formation.models.initial_conditions import ExternalTides

@dataclass
class SimulationParameters:
    Omega0: float
    e: list[float]          # [e1, e2, e3]
    ai: float               # Initial expansion factor
    zi: float               # zi = 1.0 / ai - 1.0
    delta: float            # Initial overdensity of ellipse (rho_E/rho_U - 1)
    factor: float = 1.0     # Fraction of peculiar velocity at initial epoch
    aEnd: float = 1.0       # Expansion factor at end of calc. a=1=now:
    ExternalTides: ExternalTides = ExternalTides()