# structure_formation/models/initial_conditions.py

from dataclasses import dataclass

@dataclass
class CollapseParams:
    delta: float    # Overdensity
    e: float        # Ellipticity

@dataclass
class ExternalTides:
    e11: float = 0
    e22: float = 0
    e33: float = 0