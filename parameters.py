from dataclasses import dataclass, field
import numpy as np

@dataclass
class QuadParams:
    m: float = 1.2          # Mass (kg)
    g: float = 9.81         # Gravity (m/s^2)
    l: float = 0.25         # Arm length (m)
    Inertia: np.ndarray = field(default_factory=lambda: np.diag([0.015, 0.015, 0.028]))  
    kf: float = 2.98e-6     # Thrust coefficient
    km: float = 1.14e-7     # Drag moment coefficient