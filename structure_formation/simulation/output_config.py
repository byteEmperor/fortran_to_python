# structure_formation/simulation/output_config.py

from dataclasses import dataclass

@dataclass
class OutputConfig:
    ai_path: str
    vpec_path: str
    ellipsoid_path: str
    logs_path: str = "./tmp/logs.txt" # optional default

