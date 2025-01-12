from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any
from computepropeller import Propeller


class ComputeHoverForce(om.ExplicitComponent):
    """Computes force required for hover."""
    
    def setup(self):
        self.add_input("mass_aircraft", val=1.0, units="kg",
                      desc="Total aircraft mass")
        
        self.add_output("f_hover", units="N",
                       desc="Force required for hover")
    
    def setup_partials(self):
        self.declare_partials("f_hover", "mass_aircraft", method="exact")
        
    def compute(self, inputs, outputs):
        g = 9.81
        outputs["f_hover"] = inputs["mass_aircraft"] * g
        
    def compute_partials(self, inputs, partials):
        g = 9.81
        partials["f_hover", "mass_aircraft"] = g


class ComputeHoverPowerEnergy(om.ExplicitComponent):
    """Computes power and energy requirements from propulsive power."""
    
    def setup(self):
        self.add_input("p_shaft_unit", val=1.0, units="W",
                      desc="Unit shaft power required")
        self.add_input("t_hover", val=1.0, units="s",
                      desc="Time spent in hover")
        self.add_input("hy_ratio_hover", val=1.0, units=None,
                      desc="Hover power hybridization ratio (bat fraction)")
        self.add_input("n_lift_motors", val=1.0, units=None,
                      desc="Number of lift motors")
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
        
    def setup_partials(self):
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):

        shaft_power_total = inputs["p_shaft_unit"] * inputs["n_lift_motors"]
        
        # Compute electrical power required (accounting for motor efficiencies)
        electrical_power = shaft_power_total / (
            inputs["eta_motor"] * inputs["eta_cable"] * inputs["eta_pe"]
        )
        
        # Split power between bat and generator based on hybridization ratio
        p_bat = electrical_power * inputs["hy_ratio_hover"]
        generator_power_total = electrical_power * (1 - inputs["hy_ratio_hover"])
        
        # Compute bat energy required
        outputs["e_bat"] = p_bat * inputs["t_hover"]
        
        # Compute maximum power requirements
        outputs["p_bat"] = p_bat
        outputs["p_gen_unit"] = generator_power_total / inputs["n_gens"]
        outputs["p_turbine_unit"] = generator_power_total / inputs["eta_gen"] / inputs["n_gens"] # number of turbines equals number of gens
        
    
    def compute_partials(self, inputs, partials):
        
        # Compute intermediate values
        shaft_power_total = inputs["p_shaft_unit"] * inputs["n_lift_motors"]
        eta_elec = inputs["eta_motor"] * inputs["eta_cable"] * inputs["eta_pe"]
        electrical_power = shaft_power_total / eta_elec
        p_bat = electrical_power * inputs["hy_ratio_hover"]
        p_gen_total = electrical_power * (1 - inputs["hy_ratio_hover"])
        
        # Partial derivatives for e_bat
        partials["e_bat", "p_shaft_unit"] = inputs["t_hover"] * inputs["hy_ratio_hover"] / (
            eta_elec)
        partials["e_bat", "t_hover"] = p_bat
        partials["e_bat", "hy_ratio_hover"] = electrical_power * inputs["t_hover"]
        partials["e_bat", "eta_motor"] = -p_bat * inputs["t_hover"] / inputs["eta_motor"] 
        partials["e_bat", "eta_cable"] = -p_bat * inputs["t_hover"] / inputs["eta_cable"] 
        partials["e_bat", "eta_pe"] = -p_bat * inputs["t_hover"] / inputs["eta_pe"]
        partials["e_bat", "eta_gen"] = 0.0
        partials["e_bat", "n_lift_motors"] = p_bat * inputs["t_hover"] / inputs["n_lift_motors"]
        partials["e_bat", "n_gens"] = 0.0
        
        # Partial derivatives for p_bat
        partials["p_bat", "p_shaft_unit"] = inputs["hy_ratio_hover"] * inputs["n_lift_motors"] / (
            eta_elec)
        partials["p_bat", "hy_ratio_hover"] = electrical_power
        partials["p_bat", "eta_motor"] = -p_bat / inputs["eta_motor"]
        partials["p_bat", "eta_cable"] = -p_bat / inputs["eta_cable"]
        partials["p_bat", "eta_pe"] = -p_bat / inputs["eta_pe"]
        partials["p_bat", "t_hover"] = 0.0
        partials["p_bat", "eta_gen"] = 0.0
        partials["p_bat", "n_lift_motors"] = p_bat / inputs["n_lift_motors"]
        partials["p_bat", "n_gens"] = 0.0
        
        # Partial derivatives for p_gen_unit
        partials["p_gen_unit", "p_shaft_unit"] = (1 - inputs["hy_ratio_hover"]) * inputs["n_lift_motors"] / (
            eta_elec * inputs["n_gens"])
        partials["p_gen_unit", "hy_ratio_hover"] = -electrical_power / inputs["n_gens"]
        partials["p_gen_unit", "eta_motor"] = -p_gen_total / inputs["eta_motor"] / inputs["n_gens"]
        partials["p_gen_unit", "eta_cable"] = -p_gen_total / inputs["eta_cable"] / inputs["n_gens"]
        partials["p_gen_unit", "eta_pe"] = -p_gen_total / inputs["eta_pe"] / inputs["n_gens"]
        partials["p_gen_unit", "n_gens"] = -p_gen_total / inputs["n_gens"] **2
        partials["p_gen_unit", "t_hover"] = 0.0
        partials["p_gen_unit", "eta_gen"] = 0.0
        partials["p_gen_unit", "n_lift_motors"] = p_gen_total / inputs["n_lift_motors"] / inputs["n_gens"]
        
        # Partial derivatives for gasturbine_power
        partials["p_turbine_unit", "p_shaft_unit"] = (1 - inputs["hy_ratio_hover"]) * inputs["n_lift_motors"]  / (
            eta_elec * inputs["eta_gen"]  * inputs["n_gens"])
        partials["p_turbine_unit", "hy_ratio_hover"] = -electrical_power / (inputs["eta_gen"] * inputs["n_gens"])
        partials["p_turbine_unit", "eta_motor"] = -p_gen_total / (inputs["eta_motor"] * inputs["eta_gen"] * inputs["n_gens"]   )
        partials["p_turbine_unit", "eta_cable"] = -p_gen_total / (inputs["eta_cable"] * inputs["eta_gen"] * inputs["n_gens"]   )
        partials["p_turbine_unit", "eta_pe"] = -p_gen_total / (inputs["eta_pe"] * inputs["eta_gen"] * inputs["n_gens"]   )
        partials["p_turbine_unit", "n_gens"] = -p_gen_total / inputs["eta_gen"] / inputs["n_gens"] **2
        partials["p_turbine_unit", "eta_gen"] = -p_gen_total / inputs["eta_gen"] / inputs["n_gens"]
        partials["p_turbine_unit", "t_hover"] = 0.0
        partials["p_turbine_unit", "n_lift_motors"] = p_gen_total / inputs["n_lift_motors"] / inputs["eta_gen"] / inputs["n_gens"]


class HoverEnergyGroup(om.Group):
    """Group containing hover energy analysis components."""
    
    def setup(self):

        # Add force computation
        self.add_subsystem("f_hover",
                          ComputeHoverForce(),
                          promotes_inputs=["mass_aircraft"],
                          promotes_outputs=["f_hover"])
        
        # Add propeller performance (from propeller.py)
        self.add_subsystem("propeller",
                          Propeller(),
                          promotes_inputs=["*"], 
                          promotes_outputs=["p_shaft_unit"])
        
        # Add power and energy computation
        self.add_subsystem("power_energy",
                          ComputeHoverPowerEnergy(),
                          promotes_inputs=["t_hover", "hy_ratio_hover" ,"n_lift_motors","p_shaft_unit",
                                         "n_gens", "eta_motor", "eta_cable",
                                         "eta_pe", "eta_gen"],
                          promotes_outputs=["*"])
        
        # Connect hover force to propeller thrust_total
        self.connect("f_hover", "thrust_total")
        #self.connect("n_lift_motors", "n_motors")


        # Add nonlinear solver
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options['maxiter'] = 50
        
        # Add linear solver for calculating derivatives
        self.linear_solver = om.DirectSolver()

if __name__ == "__main__":
    # Create problem
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output("mass_aircraft", val=2000.0, units="kg")
    ivc.add_output("t_hover", val=120.0, units="s")
    ivc.add_output("hy_ratio_hover", val=0.4)
    ivc.add_output("n_lift_motors", val=8.0)
    ivc.add_output("n_gens", val=2.0)
    ivc.add_output("eta_motor", val=0.95)
    ivc.add_output("d_blades", val=3.0, units="m")
    ivc.add_output("d_hub", val=1.0, units="m")
    ivc.add_output("vel", val=1e-9, units="m/s")
    ivc.add_output("eta_cable", val=0.98)
    ivc.add_output("eta_pe", val=0.95)
    ivc.add_output("eta_gen", val=0.95)
    ivc.add_output("eta_prop", val=0.8)
    ivc.add_output("eta_hover", val=0.5)
    ivc.add_output("rho", val=1.225, units="kg/m**3")
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.connect("n_lift_motors", "n_motors")
    model.add_subsystem('hover', HoverEnergyGroup(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Generate N2 diagram
    # om.n2(prob)
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nHover Energy Results:')
    print('Hover Force Required:', prob.get_val('hover.f_hover')[0], 'N')
    print('Total Propulsive Power:', prob.get_val('hover.p_shaft_unit')[0] * prob.get_val('hover.n_motors')[0] /1000, 'kW')
    print('bat Energy Required:', prob.get_val('hover.e_bat')[0]/3.6e6, 'kWh')
    print('bat Power:', prob.get_val('hover.p_bat')[0]/1000, 'kW')
    print('Generator Power (each):', prob.get_val('hover.p_gen_unit')[0]/1000, 'kW')
    print('Gas Turbine Power (each):', prob.get_val('hover.p_turbine_unit')[0]/1000, 'kW')
    
    # Check partials
    prob.check_partials(compact_print=True) 