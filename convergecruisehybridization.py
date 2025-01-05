from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any


class ConvergeCruiseHybridization(om.ImplicitComponent):
    """
    Solves for the hybridization ratio (epsilon) by comparing target and actual range.
    """

    def setup(self):
        # Add inputs
        self.add_input("target_range", val=1.0, units="m",
                      desc="Target range")
        self.add_input("actual_range", val=1.0, units="m",
                      desc="Actual range computed")

        # Add state variable (epsilon)
        self.add_output("epsilon", val=0.3, units=None,
                       desc="Fraction of energy provided by batteries",
                       lower=0.0, upper=1.0)

        # Declare partials
        self.declare_partials("epsilon", ["target_range", "actual_range"], method="exact")

    def apply_nonlinear(self, inputs, outputs, residuals):
        """Compute residual as difference between target and actual range."""
        residuals["epsilon"] = inputs["target_range"] - inputs["actual_range"]

    def linearize(self, inputs, outputs, partials):
        """Compute the partial derivatives."""
        partials["epsilon", "target_range"] = 1.0
        partials["epsilon", "actual_range"] = -1.0


if __name__ == "__main__":
    import openmdao.api as om

    # Build the model
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output("target_range", val=1000000.0, units="m")
    ivc.add_output("actual_range", val=900000.0, units="m")  # Example value

    # Add components to model
    prob.model.add_subsystem('init_vals', ivc, promotes_outputs=["*"])
    prob.model.add_subsystem('cruise_hybridization', ConvergeCruiseHybridization(), promotes_inputs=["*"])

    # Setup and run problem
    prob.setup()
    prob.run_model()

    # Print the outputs
    print('Hybridization Ratio (epsilon):', prob.get_val('cruise_hybridization.epsilon')[0]) 