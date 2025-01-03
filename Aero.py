from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

class Aero(Discipline):
    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_names([
            "AR",           # Aspect Ratio - wingspan^2 / wing area
            "S",            # Wing Area [m^2]
            "e",            # Oswald Number
            "vel",          # Velocity [m/s]
            "rho",          # Air Density [kg/m^3]
            "weight",       # Aircraft Weight [N]
            "CD0"          # Parasitic Drag Coefficient
        ])
        
        self.output_grammar.update_from_names([
            "CL",          # Lift Coefficient
            "CD",          # Total Drag Coefficient
            "drag",        # Total Drag Force [N]
            "lift"         # Total Lift Force [N]
        ])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        # Get inputs
        AR = input_data["AR"]        # Aspect Ratio
        S = input_data["S"]          # Wing Area [m^2]
        e = input_data["e"]          # Oswald Number
        vel = input_data["vel"]      # Velocity [m/s]
        rho = input_data["rho"]      # Air Density [kg/m^3]
        weight = input_data["weight"] # Aircraft Weight [N]
        CD0 = input_data["CD0"]      # Parasitic Drag Coefficient

        # Calculate CL
        CL = (2 * weight) / (rho * vel**2 * S)

        # Calculate CDi and total CD
        CDi = CL**2 / (np.pi * e * AR)
        CD = CD0 + CDi

        # Calculate forces
        q = 0.5 * rho * vel**2      # Dynamic pressure
        drag = q * S * CD
        lift = q * S * CL

        return {
            "CL": CL,
            "CD": CD,
            "drag": drag,
            "lift": lift
        }


if __name__ == "__main__":
    # Create instance of the discipline
    aero_discipline = Aero()

    # Create test input data
    test_inputs = {
        "AR": 8.0,         # Aspect Ratio
        "S": 10.0,         # Wing Area [m^2]
        "e": 0.85,         # Oswald Number
        "vel": 20.0,       # Velocity [m/s]
        "rho": 1.225,      # Air Density [kg/m^3]
        "weight": 1000.0,  # Aircraft Weight [N]
        "CD0": 0.015      # Parasitic Drag Coefficient
    }

    # Execute discipline
    result = aero_discipline._run(test_inputs)
    
    print("Aero Test Results:")
    print(f"Input conditions: {test_inputs}")
    print(f"Lift Coefficient (CL): {result['CL']:.3f}")
    print(f"Drag Coefficient (CD): {result['CD']:.3f}")
    print(f"Lift force: {result['lift']:.2f} N")
    print(f"Drag force: {result['drag']:.2f} N") 