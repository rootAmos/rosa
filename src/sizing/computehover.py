from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any
from computepropeller import HoverPropeller




class ComputeHover(om.ExplicitComponent):
    """Computes power and energy requirements from propulsive power."""
    
    def setup(self):
        self.add_input("p_shaft_unit", val=1.0, units="W",
                      desc="Unit shaft power required")
        self.add_input("t_hover", val=1.0, units="s",
                      desc="Time spent in hover")
        self.add_input("epsilon_hvr", val=1.0, units=None,
                      desc="Hover power hybridization ratio (bat fraction)")
        self.add_input("n_lift_props", val=1.0, units=None,
                      desc="Number of lifting propellers")
        self.add_input("n_gens", val=1.0, units=None,
                      desc="Number of gens")
        
        # Efficiencies
        self.add_input("eta_motor", val=1.0, units=None,
                      desc="Motor efficiency")
        self.add_input("eta_cable", val=1.0, units=None,
                      desc="Cable efficiency")
        self.add_input("eta_pe", val=1.0, units=None,
                      desc="Power electronics efficiency")
        self.add_input("eta_gen", val=1.0, units=None,
                      desc="Generator efficiency")
        
        # Add outputs
        self.add_output("e_bat", units="J",
                       desc="Total bat hover energy required")
        self.add_output("p_bat", units="W",
                       desc="Maximum bat power required")
        self.add_output("p_gen_unit", units="W",
                       desc="Maximum power per generator")
        self.add_output("p_turbine_unit", units="W",
                       desc="Gas turbine power required")
        self.add_output("p_motor", units="W",
                       desc="Unit motor power required")
        self.add_output("p_elec", units="W",
                       desc="Total electrical power required")
        
    def setup_partials(self):
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):

        shaft_power_total = inputs["p_shaft_unit"] * inputs["n_lift_props"]
        
        # Compute electrical power required (accounting for motor efficiencies)
        electrical_power = shaft_power_total / (
            inputs["eta_motor"] * inputs["eta_cable"] * inputs["eta_pe"]
        )

        p_motor_unit = inputs["p_shaft_unit"] / inputs["eta_motor"] 
        
        # Split power between bat and generator based on hybridization ratio
        p_bat = electrical_power * inputs["epsilon_hvr"]
        generator_power_total = electrical_power * (1 - inputs["epsilon_hvr"])
        
        # Compute bat energy required
        outputs["e_bat"] = p_bat * inputs["t_hover"]
        
        # Compute maximum power requirements
        outputs["p_bat"] = p_bat
        outputs["p_gen_unit"] = generator_power_total / inputs["n_gens"]
        outputs["p_turbine_unit"] = generator_power_total / inputs["eta_gen"] / inputs["n_gens"] # number of turbines equals number of gens
        outputs["p_elec"] = electrical_power
        outputs["p_motor"] = p_motor_unit
    
    def compute_partials(self, inputs, partials):
        
        # Compute intermediate values
        shaft_power_total = inputs["p_shaft_unit"] * inputs["n_lift_props"]
        eta_elec = inputs["eta_motor"] * inputs["eta_cable"] * inputs["eta_pe"]
        electrical_power = shaft_power_total / eta_elec
        p_bat = electrical_power * inputs["epsilon_hvr"]
        p_gen_total = electrical_power * (1 - inputs["epsilon_hvr"])
        p_motor_unit = inputs["p_shaft_unit"] / inputs["eta_motor"] 

        
        # Partial derivatives for e_bat
        partials["e_bat", "p_shaft_unit"] = inputs["t_hover"] * inputs["epsilon_hvr"] / (
            eta_elec)
        partials["e_bat", "t_hover"] = p_bat
        partials["e_bat", "epsilon_hvr"] = electrical_power * inputs["t_hover"]
        partials["e_bat", "eta_motor"] = -p_bat * inputs["t_hover"] / inputs["eta_motor"] 
        partials["e_bat", "eta_cable"] = -p_bat * inputs["t_hover"] / inputs["eta_cable"] 
        partials["e_bat", "eta_pe"] = -p_bat * inputs["t_hover"] / inputs["eta_pe"]
        partials["e_bat", "eta_gen"] = 0.0
        partials["e_bat", "n_lift_props"] = p_bat * inputs["t_hover"] / inputs["n_lift_props"]
        partials["e_bat", "n_gens"] = 0.0
        
        # Partial derivatives for p_bat
        partials["p_bat", "p_shaft_unit"] = inputs["epsilon_hvr"] * inputs["n_lift_props"] / (
            eta_elec)
        partials["p_bat", "epsilon_hvr"] = electrical_power
        partials["p_bat", "eta_motor"] = -p_bat / inputs["eta_motor"]
        partials["p_bat", "eta_cable"] = -p_bat / inputs["eta_cable"]
        partials["p_bat", "eta_pe"] = -p_bat / inputs["eta_pe"]
        partials["p_bat", "t_hover"] = 0.0
        partials["p_bat", "eta_gen"] = 0.0
        partials["p_bat", "n_lift_props"] = p_bat / inputs["n_lift_props"]
        partials["p_bat", "n_gens"] = 0.0
        
        # Partial derivatives for p_gen_unit
        partials["p_gen_unit", "p_shaft_unit"] = (1 - inputs["epsilon_hvr"]) * inputs["n_lift_props"] / (
            eta_elec * inputs["n_gens"])
        partials["p_gen_unit", "epsilon_hvr"] = -electrical_power / inputs["n_gens"]
        partials["p_gen_unit", "eta_motor"] = -p_gen_total / inputs["eta_motor"] / inputs["n_gens"]
        partials["p_gen_unit", "eta_cable"] = -p_gen_total / inputs["eta_cable"] / inputs["n_gens"]
        partials["p_gen_unit", "eta_pe"] = -p_gen_total / inputs["eta_pe"] / inputs["n_gens"]
        partials["p_gen_unit", "n_gens"] = -p_gen_total / inputs["n_gens"] **2
        partials["p_gen_unit", "t_hover"] = 0.0
        partials["p_gen_unit", "eta_gen"] = 0.0
        partials["p_gen_unit", "n_lift_props"] = p_gen_total / inputs["n_lift_props"] / inputs["n_gens"]
        
        # Partial derivatives for gasturbine_power
        partials["p_turbine_unit", "p_shaft_unit"] = (1 - inputs["epsilon_hvr"]) * inputs["n_lift_props"]  / (
            eta_elec * inputs["eta_gen"]  * inputs["n_gens"])
        partials["p_turbine_unit", "epsilon_hvr"] = -electrical_power / (inputs["eta_gen"] * inputs["n_gens"])
        partials["p_turbine_unit", "eta_motor"] = -p_gen_total / (inputs["eta_motor"] * inputs["eta_gen"] * inputs["n_gens"]   )
        partials["p_turbine_unit", "eta_cable"] = -p_gen_total / (inputs["eta_cable"] * inputs["eta_gen"] * inputs["n_gens"]   )
        partials["p_turbine_unit", "eta_pe"] = -p_gen_total / (inputs["eta_pe"] * inputs["eta_gen"] * inputs["n_gens"]   )
        partials["p_turbine_unit", "n_gens"] = -p_gen_total / inputs["eta_gen"] / inputs["n_gens"] **2
        partials["p_turbine_unit", "eta_gen"] = -p_gen_total / inputs["eta_gen"] / inputs["n_gens"]
        partials["p_turbine_unit", "t_hover"] = 0.0
        partials["p_turbine_unit", "n_lift_props"] = p_gen_total / inputs["n_lift_props"] / inputs["eta_gen"] / inputs["n_gens"]


        partials["p_motor", "p_shaft_unit"] = 1 / inputs["eta_motor"]
        partials["p_motor", "eta_motor"] = -p_motor_unit / inputs["eta_motor"]
        partials["p_motor", "eta_cable"] = 0.0
        partials["p_motor", "eta_pe"] = 0.0
        partials["p_motor", "eta_gen"] = 0.0
        partials["p_motor", "n_lift_props"] = 0.0
        partials["p_motor", "n_gens"] = 0.0

        partials["p_elec", "p_shaft_unit"] = 1 / eta_elec
        partials["p_elec", "eta_motor"] = -electrical_power / inputs["eta_motor"]
        partials["p_elec", "eta_cable"] = -electrical_power / inputs["eta_cable"]
        partials["p_elec", "eta_pe"] = -electrical_power / inputs["eta_pe"]
        partials["p_elec", "eta_gen"] = 0.0
        partials["p_elec", "n_lift_props"] = 0.0
        partials["p_elec", "n_gens"] = 0.0

class HoverEnergyAnalysis(om.Group):
    """Group containing hover energy analysis components."""
    
    def setup(self):

        # Add propeller performance (from propeller.py)
        self.add_subsystem("hoverpropeller",
                          HoverPropeller(),
                          promotes_inputs=["rho","eta_prop","eta_hover"], 
                          promotes_outputs=[])
        
        # Add power and energy computation
        self.add_subsystem("computehover",
                          ComputeHover(),
                          promotes_inputs=["t_hover", "epsilon_hvr" ,
                                         "n_gens", "eta_motor", "eta_cable",
                                         "eta_pe", "eta_gen"],
                          promotes_outputs=["*"])
        
        # Connect hover force to propeller thrust_total
        self.connect("hoverpropeller.p_shaft_unit", "computehover.p_shaft_unit")


        # Add nonlinear solver
        #self.nonlinear_solver = om.NonlinearBlockGS()
        #self.nonlinear_solver.options['maxiter'] = 50
        
        # Add linear solver for calculating derivatives
        #self.linear_solver = om.DirectSolver()

if __name__ == "__main__":
    # Create problem
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output("w_mto", val=5700*9.806, units="N")
    ivc.add_output("t_hover", val=120.0, units="s")
    ivc.add_output("epsilon_hvr", val=0.4)
    ivc.add_output("n_lift_props", val=8.0)
    ivc.add_output("n_gens", val=2.0)
    ivc.add_output("eta_motor", val=0.95)
    ivc.add_output("d_blades", val=3.0, units="m")
    ivc.add_output("d_hub", val=0.45, units="m")
    ivc.add_output("eta_cable", val=0.98)
    ivc.add_output("eta_pe", val=0.95)
    ivc.add_output("eta_gen", val=0.95)
    ivc.add_output("eta_prop", val=0.8)
    ivc.add_output("eta_hover", val=0.8)
    ivc.add_output("rho", val=1.225, units="kg/m**3")
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.connect("n_lift_props", "hoverpropeller.n_motors")
    model.connect("w_mto", "hoverpropeller.thrust_total")
    model.connect("d_blades", "hoverpropeller.d_blades")
    model.connect("d_hub", "hoverpropeller.d_hub")
    model.connect("n_lift_props", "computehover.n_lift_props")
    model.add_subsystem('hover', HoverEnergyAnalysis(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Generate N2 diagram
    # om.n2(prob)
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nHover Energy Results:')
    print('Total Propulsive Power:', prob.get_val('hover.hoverpropeller.p_shaft_unit')[0] * prob.get_val('hover.hoverpropeller.n_motors')[0] /1000, 'kW')
    print('bat Energy Required:', prob.get_val('hover.e_bat')[0]/3.6e6, 'kWh')
    print('bat Power:', prob.get_val('hover.p_bat')[0]/1000, 'kW')
    print('Generator Power (each):', prob.get_val('hover.p_gen_unit')[0]/1000, 'kW')
    print('Gas Turbine Power (each):', prob.get_val('hover.p_turbine_unit')[0]/1000, 'kW')
    
    # Check partials
    #prob.check_partials(compact_print=True) 