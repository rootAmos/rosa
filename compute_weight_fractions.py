from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any


class WeightFractions(om.ExplicitComponent):
    """
    Computes empty weight fraction and other weight ratios excluding payload and batteries.
    """
    
    def setup(self):
        # Structural weights
        self.add_input("w_wing", val=0.0, units="N",
                      desc="Wing weight")
        self.add_input("w_fuselage", val=0.0, units="N",
                      desc="Fuselage weight")
        self.add_input("w_lg", val=0.0, units="N",
                      desc="Landing gear weight")
        self.add_input("w_nacelle", val=0.0, units="N",
                      desc="Nacelle weight")
        
        # Propulsion weights (excluding battery)
        self.add_input("w_motor", val=0.0, units="N",
                      desc="Total motor weight")
        self.add_input("w_power_electronics", val=0.0, units="N",
                      desc="Power electronics weight")
        self.add_input("w_cables", val=0.0, units="N",
                      desc="Cable weight")
        self.add_input("w_turbine", val=0.0, units="N",
                      desc="Gas turbine weight")
        self.add_input("w_generator", val=0.0, units="N",
                      desc="Generator weight")
        
        # Systems weight
        self.add_input("w_systems", val=0.0, units="N",
                      desc="Systems weight")
        
        # Total weight for fractions
        self.add_input("w_mtow", val=0.0, units="N",
                      desc="Maximum takeoff weight")
        
        # Add outputs
        self.add_output("w_empty", units="N",
                       desc="Empty weight (excluding battery)")
        self.add_output("empty_weight_fraction", units=None,
                       desc="Empty weight fraction (excluding battery)")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):
        # Sum all weights excluding battery
        outputs["w_empty"] = (inputs["w_wing"] + inputs["w_fuselage"] + 
                            inputs["w_lg"] + inputs["w_nacelle"] +
                            inputs["w_motor"] + inputs["w_power_electronics"] +
                            inputs["w_cables"] + inputs["w_turbine"] +
                            inputs["w_generator"] + inputs["w_systems"])
        
        # Compute empty weight fraction
        outputs["empty_weight_fraction"] = outputs["w_empty"] / inputs["w_mtow"]
        
    def compute_partials(self, inputs, partials):
        # Partials for w_empty
        weight_components = ["w_wing", "w_fuselage", "w_lg", "w_nacelle",
                           "w_motor", "w_power_electronics", "w_cables",
                           "w_turbine", "w_generator", "w_systems"]
        
        for component in weight_components:
            partials["w_empty", component] = 1.0
        
        # Partials for empty_weight_fraction
        w_empty = sum(inputs[component] for component in weight_components)
        
        for component in weight_components:
            partials["empty_weight_fraction", component] = 1.0 / inputs["w_mtow"]
        
        partials["empty_weight_fraction", "w_mtow"] = -w_empty / inputs["w_mtow"]**2


if __name__ == "__main__":
    # Create problem
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    
    # Add structural weights
    ivc.add_output("w_wing", val=5000.0, units="N")
    ivc.add_output("w_fuselage", val=3000.0, units="N")
    ivc.add_output("w_lg", val=1000.0, units="N")
    ivc.add_output("w_nacelle", val=500.0, units="N")
    
    # Add propulsion weights
    ivc.add_output("w_motor", val=1000.0, units="N")
    ivc.add_output("w_power_electronics", val=500.0, units="N")
    ivc.add_output("w_cables", val=300.0, units="N")
    ivc.add_output("w_turbine", val=2000.0, units="N")
    ivc.add_output("w_generator", val=1000.0, units="N")
    
    # Add systems weight
    ivc.add_output("w_systems", val=1500.0, units="N")
    
    # Add total weight
    ivc.add_output("w_mtow", val=20000.0, units="N")
    
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
    print('Empty Weight Fraction:', prob.get_val('weight_fractions.empty_weight_fraction')[0])
    
    # Check partials
    prob.check_partials(compact_print=True) 