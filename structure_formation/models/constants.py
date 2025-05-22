# structure_formation/models/constants.py

from dataclasses import dataclass

@dataclass
class UniverseParams:
    Omega0: float   # Matter density parameter today
    ai: float       # Initial scale factor
    zi: float       # Initial redshift
