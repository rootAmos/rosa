from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any


class WeightFractions(om.ExplicitComponent):
    """
    Computes weight fractions needed for cruise range estimation.
    """
    
    def setup(self):
        self.add_input("w_pay", val=0.0, units="N",
                      desc="Payload weight")
        
        self.add_input("w_bat", val=0.0, units="N",
                      desc="bat weight")     
        
        self.add_input("w_mto", val=0.0, units="N",
                      desc="Maximum takeoff weight")

        self.add_output("f_we", units=None,
                       desc="Empty weight fraction (excluding bat)")
        self.add_input("w_fuel", val=0.0, units="N",
                      desc="fuel weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):

        # Unpack inputs
        w_pay = inputs["w_pay"]
        w_bat = inputs["w_bat"]
        w_mto = inputs["w_mto"]
        epsilon_crz = inputs["epsilon_crz"]

        w_fuel = w_bat * (1 - epsilon_crz) / epsilon_crz

        w_empty = w_mto - w_pay - w_bat - w_fuel
        
        # Compute empty weight fraction
        outputs["f_we"] = w_empty / w_mto
        outputs["w_fuel"] = w_fuel
        
    def compute_partials(self, inputs, partials):

        # Unpack inputs 
        w_mto = inputs["w_mto"]
        w_empty = inputs["w_empty"]
        w_bat = inputs["w_bat"]
        epsilon_crz = inputs["epsilon_crz"]
        w_pay = inputs["w_pay"]

        w_fuel = w_bat * (1 - epsilon_crz) / epsilon_crz

        w_empty = w_mto - w_pay - w_bat - w_fuel

        # Compute partials
        partials["f_we", "w_mto"] = (w_bat - w_mto + w_pay - (w_bat*(epsilon_crz - 1))/epsilon_crz)/w_mto^2 + 1/w_mto

        partials["f_we", "w_pay"] = -1/w_mto

        partials["f_we", "w_bat"] = ((epsilon_crz - 1)/epsilon_crz - 1)/w_mto

        partials["f_we", "epsilon_crz"] = (w_bat/epsilon_crz - (w_bat*(epsilon_crz - 1))/epsilon_crz^2)/w_mto


        # Partials for w_fuel
        partials["w_fuel", "w_bat"] = -(epsilon_crz - 1)/epsilon_crz

        partials["w_fuel", "epsilon_crz"] = (w_bat*(epsilon_crz - 1))/epsilon_crz^2 - w_bat/epsilon_crz

        partials["w_fuel", "w_mto"] = 0.0
        partials["w_fuel", "w_wing"] = 0.0
        partials["w_fuel", "w_fuselage"] = 0.0
        partials["w_fuel", "w_lg"] = 0.0
        partials["w_fuel", "w_nacelles"] = 0.0
        partials["w_fuel", "w_ptrain"] = 0.0
        partials["w_fuel", "w_systems"] = 0.0


if __name__ == "__main__":
    # Create problem
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    
    # Add structural weights
    ivc.add_output("w_wing", val=5000.0, units="N")
    ivc.add_output("w_fuselage", val=3000.0, units="N")
    ivc.add_output("w_lg", val=1000.0, units="N")
    ivc.add_output("w_nacelles", val=500.0, units="N")
    
    # Add propulsion weights
    ivc.add_output("w_ptrain", val=20000.0, units="N")
    ivc.add_output("w_bat", val=1000.0, units="N")
    ivc.add_output("w_fuel", val=1000.0, units="N")

    # Add systems weight
    ivc.add_output("w_systems", val=1500.0, units="N")
    
    # Add total weight
    ivc.add_output("w_mto", val=40000.0, units="N")
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('weight_fractions', WeightFractions(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nWeight Fraction Results:')
    print('Empty Weight:', prob.get_val('weight_fractions.w_empty')[0], 'N')
    print('Empty Weight Fraction:', prob.get_val('weight_fractions.f_we')[0])
    
    # Check partials
    prob.check_partials(compact_print=True) 