from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any


class ComputeCruiseRange(om.ExplicitComponent):
    """
    Computes the total range of a hybrid-electric aircraft based on the Breguet range equation.
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
        self.add_input("epsilon", val=1.0, units=None,
                      desc="Fraction of energy provided by batteries")
        self.add_input("cb", val=1.0, units="J/kg",
                      desc="Battery specific energy")
        self.add_input("cp", val=1.0, units="1/s",
                      desc="Shaft power specific fuel consumption")
        self.add_input("wt0_max", val=1.0, units="N",
                      desc="Maximum takeoff weight")
        self.add_input("w_pay", val=1.0, units="N",
                      desc="Payload weight")
        self.add_input("f_we", val=1.0, units=None,
                      desc="Empty weight fraction")

        # Add output
        self.add_output("range_total", units="m",
                       desc="Total range")

        # Declare partials
        self.declare_partials("range_total", ["epsilon"], method="exact")
        self.declare_partials("range_total", ["cl", "cd", "eta_i", "eta_m", "eta_p", 
                                            "eta_g", "cb", "cp", "wt0_max", "w_pay", 
                                            "f_we"], method="fd")

    def compute(self, inputs, outputs):
        """Compute the total range."""
        g = 9.806
        
        # Get inputs
        cl = inputs["cl"]
        cd = inputs["cd"]
        eta_i = inputs["eta_i"]
        eta_m = inputs["eta_m"]
        eta_p = inputs["eta_p"]
        eta_g = inputs["eta_g"]
        epsilon = inputs["epsilon"]
        cb = inputs["cb"]
        cp = inputs["cp"]
        wt0_max = inputs["wt0_max"]
        w_pay = inputs["w_pay"]
        f_we = inputs["f_we"]

        outputs["range_total"] = (
            (cl / cd) * (eta_i * eta_m * eta_p / g)
            * np.log(
                ((epsilon + (1 - epsilon) * cb * cp) * wt0_max)
                / (epsilon * wt0_max + (1 - epsilon) * cb * cp * (w_pay + f_we * wt0_max))
            ) * eta_g / cp
            + (epsilon * cb * ((1 - f_we) * wt0_max - w_pay))
            / (epsilon * wt0_max + (1 - epsilon) * cb * cp * (w_pay + f_we * wt0_max))
        )

    def compute_partials(self, inputs, partials):
        """Compute the partial derivative of range_total with respect to epsilon."""
        g = 9.806
        
        # Get inputs
        cl = inputs["cl"]
        cd = inputs["cd"]
        eta_i = inputs["eta_i"]
        eta_m = inputs["eta_m"]
        eta_p = inputs["eta_p"]
        eta_g = inputs["eta_g"]
        epsilon = inputs["epsilon"]
        cb = inputs["cb"]
        cp = inputs["cp"]
        wt0_max = inputs["wt0_max"]
        w_pay = inputs["w_pay"]
        f_we = inputs["f_we"]
        
        # Terms for readability
        term1 = (cl/cd) * (eta_i * eta_m * eta_p / g) * (eta_g / cp)
        term2 = wt0_max * (1 - cb * cp)
        term3 = w_pay + f_we * wt0_max
        term4 = epsilon * wt0_max + (1 - epsilon) * cb * cp * term3
        
        # Derivative of the logarithmic term
        d_log = (wt0_max * cb * cp * term3) / (term4 * (epsilon * wt0_max + (1 - epsilon) * cb * cp * term3))
        
        # Derivative of the second term
        d_second = (cb * ((1-f_we) * wt0_max - w_pay) * wt0_max * (1 - cb * cp)) / (term4 ** 2)
        
        partials["range_total", "epsilon"] = term1 * d_log + d_second


if __name__ == "__main__":
    import openmdao.api as om

    # Build the model
    prob = om.Problem()
    
    # Create independent variable component with units
    ivc = om.IndepVarComp()
    ivc.add_output("cl", val=0.5, units=None)
    ivc.add_output("cd", val=0.03, units=None)
    ivc.add_output("eta_i", val=0.95, units=None)
    ivc.add_output("eta_m", val=0.95, units=None)
    ivc.add_output("eta_p", val=0.85, units=None)
    ivc.add_output("eta_g", val=0.95, units=None)
    ivc.add_output("epsilon", val=0.3, units=None)
    ivc.add_output("cb", val=400*3600, units="J/kg")
    ivc.add_output("cp", val=0.3/60/60/1000, units="1/s")
    ivc.add_output("wt0_max", val=5500.0*9.806, units="N")
    ivc.add_output("w_pay", val=200.0*9.806, units="N")
    ivc.add_output("f_we", val=0.6, units=None)

    # Add components to model
    prob.model.add_subsystem('init_vals', ivc, promotes_outputs=["*"])
    prob.model.add_subsystem('cruise_range', ComputeCruiseRange(), promotes_inputs=["*"])

    # Setup and run problem
    prob.setup()
    prob.run_model()

    # Print the outputs
    print('Range:', prob.get_val('cruise_range.range_total')[0]/1000, 'km') 