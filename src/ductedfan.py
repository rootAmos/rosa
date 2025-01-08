from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

class DuctedFan(Discipline):
    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_names([
            "total_thrust_req",
            "num_motors",
            "d_blade",
            "d_hub",
            "vel",
            "rho",
            "eta_prop",
            "eta_prplsv",
            "k_duct"
        ])
        
        self.output_grammar.update_from_names(["unit_shaft_pow"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        # Get inputs
        total_thrust_req = input_data["total_thrust_req"]
        num_motors = input_data["num_motors"]
        d_blade = input_data["d_blade"]
        d_hub = input_data["d_hub"]
        vel = input_data["vel"]
        rho = input_data["rho"]
        eta_prop = input_data["eta_prop"]
        eta_prplsv = input_data["eta_prplsv"]
        k_duct = input_data["k_duct"]

        # Compute unit thrust required
        unit_thrust_req = total_thrust_req / num_motors
        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        # Compute propulsive power required [1] Eq 15-75 with duct factor
        unit_propulsive_pow_req = unit_thrust_req * vel + \
            unit_thrust_req ** 1.5 / (k_duct * np.sqrt(2 * rho * diskarea))

        # Compute shaft power required [1] Eq 15-78
        unit_shaft_pow = unit_propulsive_pow_req / eta_prop / eta_prplsv

        return {"unit_shaft_pow": unit_shaft_pow}

if __name__ == "__main__":
    # Create instance of the discipline
    fan_discipline = DuctedFan()

    # Create test input data
    test_inputs = {
        "total_thrust_req": 1000.0,  # N
        "num_motors": 4,
        "d_blade": 0.5,    # m
        "d_hub": 0.05,     # m
        "vel": 20.0,       # m/s
        "rho": 1.225,      # kg/m^3
        "eta_prop": 0.8,   # efficiency
        "eta_prplsv": 0.9, # efficiency
        "k_duct": 1.2      # duct factor
    }

    # Execute discipline
    result = fan_discipline._run(test_inputs)
    
    print("Ducted Fan Test Results:")
    print(f"Input conditions: {test_inputs}")
    print(f"Unit shaft power required: {result['unit_shaft_pow']:.2f} W")