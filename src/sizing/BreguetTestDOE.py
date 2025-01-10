from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gemseo import create_scenario, configure_logger
from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import Discipline
from gemseo.settings.post import ScatterPlotMatrix_Settings  

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

# Configure logging
configure_logger()

class AircraftRangeDiscipline(Discipline):
    def __init__(self) -> None:
        super().__init__()
        # Initialize input/output grammars
        self.input_grammar.update_from_names([
            "velocity",
            "specific_fuel_consumption",
            "lift_to_drag",
            "initial_weight", 
            "final_weight"
        ])
        self.output_grammar.update_from_names(["range"])
        
        # Define default inputs
        self.default_input_data = {
            "velocity": 250.0,
            "specific_fuel_consumption": 0.8,
            "lift_to_drag": 15.0,
            "initial_weight": 15000.0,
            "final_weight": 10000.0,
        }

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        # Extract inputs
        velocity = input_data["velocity"]
        specific_fuel_consumption = input_data["specific_fuel_consumption"] 
        lift_to_drag = input_data["lift_to_drag"]
        initial_weight = input_data["initial_weight"]
        final_weight = input_data["final_weight"]

        #if initial_weight <= final_weight:
        #    raise ValueError("Initial weight must be greater than final weight.")

        # Compute range
        range_km = -(
            (velocity / specific_fuel_consumption)
            * lift_to_drag
            * np.log(initial_weight / final_weight)
            / 1000
        )

        return {"range": range_km}

# Create discipline instance
discipline = AircraftRangeDiscipline()

# Create design space
design_space = DesignSpace()
design_space.add_variable("velocity", 1, lower_bound=200.0, upper_bound=800.0, value=250.0)
design_space.add_variable("specific_fuel_consumption", 1, lower_bound=0.5, upper_bound=2.0, value=0.8)
design_space.add_variable("lift_to_drag", 1, lower_bound=10.0, upper_bound=20.0, value=15.0)
design_space.add_variable("initial_weight", 1, lower_bound=5000.0, upper_bound=20000.0, value=15000.0)
design_space.add_variable("final_weight", 1, lower_bound=3000.0, upper_bound=15000.0, value=10000.0)

# Create and execute scenario
scenario = create_scenario(
    disciplines=[discipline],
    formulation_name="DisciplinaryOpt",
    objective_name="range",
    design_space=design_space,
    #scenario_type="DOE"
)

scenario.set_differentiation_method("finite_differences")

# Execute Design of Experiments
#scenario.execute(algo_name="PYDOE_LHS", n_samples=300)

# Execute Optimization
scenario.execute(algo_name="SLSQP", max_iter=100)


# Optional: Generate plots
from gemseo.settings.post import ScatterPlotMatrix_Settings  # noqa: E402

settings_model = ScatterPlotMatrix_Settings(
    variable_names=["velocity", "specific_fuel_consumption", "lift_to_drag", "initial_weight", "final_weight", "range"],
    save=False,
    show=True,
)

scenario.post_process(settings_model)

#scenario.post_process(post_name="OptHistoryView", save=False, show=True)
