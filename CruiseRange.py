from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any


class CruiseRange(om.ImplicitComponent):
    """
    Solves for the hybridization ratio (epsilon) given a target range using the Breguet range equation.
    """

    def setup(self):
        # Add input variables (parameters) - all with val=1 and units
        self.add_input("cl", val=1.0, units=None,
                      desc="Lift coefficient")
        self.add_input("cd", val=1.0, units=None,
                      desc="Drag coefficient")
        self.add_input("eta_i", val=1.0, units=None,
                      desc="Inverter efficiency")
        self.add_input("eta_m", val=1.0, units=None,
                      desc="Motor efficiency")
        self.add_input("eta_p", val=1.0, units=None,
                      desc="Propulsive efficiency")
        self.add_input("eta_g", val=1.0, units=None,
                      desc="Generator efficiency")
        self.add_input("cb", val=1.0, units="J/kg",
                      desc="Battery specific energy")
        self.add_input("cp", val=1.0, units="kg/s/W",
                      desc="Shaft power specific fuel consumption")
        self.add_input("wto_max", val=1.0, units="N",
                      desc="Maximum takeoff weight")
        self.add_input("w_pay", val=1.0, units="N",
                      desc="Payload weight")
        self.add_input("f_we", val=1.0, units=None,
                      desc="Empty weight fraction")
        self.add_input("target_range", val=1.0, units="m",
                      desc="Target range")

        # Add state variable (epsilon) and residual
        self.add_output("epsilon", val=0.3, units=None,
                       desc="Fraction of energy provided by batteries",
                       lower=0.0, upper=1.0)
        
    def setup_partials(self):

        # Declare partials
        self.declare_partials("epsilon", "target_range", method="exact")
        self.declare_partials("epsilon", ["cl", "cd", "eta_i", "eta_m", "eta_p", 
                                               "eta_g", "cb", "cp", "wto_max", "w_pay", 
                                               "f_we"], method="fd")

    def apply_nonlinear(self, inputs, outputs, residuals):
        """Compute residuals (range_actual - target_range)."""
        g = 9.806
        
        # Get inputs and outputs
        cl = inputs["cl"]
        cd = inputs["cd"]
        eta_i = inputs["eta_i"]
        eta_m = inputs["eta_m"]
        eta_p = inputs["eta_p"]
        eta_g = inputs["eta_g"]
        epsilon = outputs["epsilon"]  # Note: epsilon is now an output
        cb = inputs["cb"]
        cp = inputs["cp"]
        wto_max = inputs["wto_max"]
        w_pay = inputs["w_pay"]
        f_we = inputs["f_we"]
        target_range = inputs["target_range"]

        # Calculate actual range
        range_actual = (
            (cl / cd) * (eta_i * eta_m * eta_p / g)
            * np.log(
                ((epsilon + (1 - epsilon) * cb * cp) * wto_max)
                / (epsilon * wto_max + (1 - epsilon) * cb * cp * (w_pay + f_we * wto_max))
            ) * eta_g / cp
            + (epsilon * cb * ((1 - f_we) * wto_max - w_pay))
            / (epsilon * wto_max + (1 - epsilon) * cb * cp * (w_pay + f_we * wto_max))
        )

        # Compute residual
        residuals["range_residual"] = range_actual - target_range

    def linearize(self, inputs, outputs, partials):
        """Compute the partial derivative of residual with respect to target_range."""
        
        # The partial of (range_actual - target_range) with respect to target_range is -1
        partials["range_residual", "target_range"] = -1.0

        # The partial with respect to epsilon remains the same
        g = 9.806
        
        # Get inputs and outputs
        cl = inputs["cl"]
        cd = inputs["cd"]
        eta_i = inputs["eta_i"]
        eta_m = inputs["eta_m"]
        eta_p = inputs["eta_p"]
        eta_g = inputs["eta_g"]
        epsilon = outputs["epsilon"]
        cb = inputs["cb"]
        cp = inputs["cp"]
        wto_max = inputs["wto_max"]
        w_pay = inputs["w_pay"]
        f_we = inputs["f_we"]
        
        # Terms for readability
        term1 = (cl/cd) * (eta_i * eta_m * eta_p / g) * (eta_g / cp)
        term2 = wto_max * (1 - cb * cp)
        term3 = w_pay + f_we * wto_max
        term4 = epsilon * wto_max + (1 - epsilon) * cb * cp * term3
        
        # Derivative of the logarithmic term
        d_log = (wto_max * cb * cp * term3) / (term4 * (epsilon * wto_max + (1 - epsilon) * cb * cp * term3))
        
        # Derivative of the second term
        d_second = (cb * ((1-f_we) * wto_max - w_pay) * wto_max * (1 - cb * cp)) / (term4 ** 2)
        
        partials["range_residual", "epsilon"] = term1 * d_log + d_second


if __name__ == "__main__":
    import openmdao.api as om

    # Build the model
    prob = om.Problem()
    
    # Create independent variable component with units
    ivc = om.IndepVarComp()
    ivc.add_output("cl", val=0.7, units=None)
    ivc.add_output("cd", val=0.03, units=None)
    ivc.add_output("eta_i", val=0.95, units=None)
    ivc.add_output("eta_m", val=0.9, units=None)
    ivc.add_output("eta_p", val=0.85, units=None)
    ivc.add_output("eta_g", val=0.92, units=None)
    ivc.add_output("cb", val=400*3600, units="J/kg")
    ivc.add_output("cp", val=0.3/60/60/1000, units="kg/s/W")
    ivc.add_output("wto_max", val=5500.0*9.806, units="N")
    ivc.add_output("w_pay", val=200.0*9.806, units="N")
    ivc.add_output("f_we", val=0.5, units=None)
    ivc.add_output("target_range", val=2725*1000.0, units="m")

    # Add components to model
    prob.model.add_subsystem('init_vals', ivc, promotes_outputs=["*"])
    prob.model.add_subsystem('cruise_range', CruiseRange(), promotes_inputs=["*"])

    # Setup and run problem
    prob.setup()
    prob.run_model()

    # Print the outputs (only epsilon, not residuals)
    print('Hybridization Ratio (epsilon):', prob.get_val('cruise_range.epsilon')[0])
