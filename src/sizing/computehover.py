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
        
        self.add_output("hover_force", units="N",
                       desc="Force required for hover")
    
    def setup_partials(self):
        self.declare_partials("hover_force", "mass_aircraft", method="exact")
        
    def compute(self, inputs, outputs):
        g = 9.81
        outputs["hover_force"] = inputs["mass_aircraft"] * g
        
    def compute_partials(self, inputs, partials):
        g = 9.81
        partials["hover_force", "mass_aircraft"] = g


class ComputeHoverPowerEnergy(om.ExplicitComponent):
    """Computes power and energy requirements from propulsive power."""
    
    def setup(self):
        self.add_input("shaft_power_unit", val=1.0, units="W",
                      desc="Unit shaft power required")
        self.add_input("hover_time", val=1.0, units="s",
                      desc="Time spent in hover")
        self.add_input("epsilon_hover", val=1.0, units=None,
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
        self.add_input("eta_electronics", val=1.0, units=None,
                      desc="Power electronics efficiency")
        self.add_input("eta_generator", val=1.0, units=None,
                      desc="Generator efficiency")
        
        # Add outputs
        self.add_output("bat_energy", units="J",
                       desc="Total bat energy required")
        self.add_output("bat_power", units="W",
                       desc="Maximum bat power required")
        self.add_output("lift_motor_power_unit", units="W",
                       desc="Maximum power per lift motor")
        self.add_output("generator_power_unit", units="W",
                       desc="Maximum power per generator")
        self.add_output("gasturbine_power_total", units="W",
                       desc="Gas turbine power required")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):

        shaft_power_total = inputs["shaft_power_unit"] * inputs["n_lift_motors"]
        
        # Compute electrical power required (accounting for motor efficiencies)
        electrical_power = shaft_power_total / (
            inputs["eta_motor"] * inputs["eta_cable"] * inputs["eta_electronics"]
        )
        
        # Split power between bat and generator based on hybridization ratio
        bat_power = electrical_power * inputs["epsilon_hover"]
        generator_power_total = electrical_power * (1 - inputs["epsilon_hover"])
        
        # Compute bat energy required
        outputs["bat_energy"] = bat_power * inputs["hover_time"]
        
        # Compute maximum power requirements
        outputs["bat_power"] = bat_power
        outputs["lift_motor_power_unit"] = shaft_power_total / inputs["n_lift_motors"]
        outputs["generator_power_unit"] = generator_power_total / inputs["n_gens"]
        outputs["gasturbine_power_total"] = outputs["generator_power_unit"]  * inputs["n_gens"] / (
            inputs["eta_generator"] 
        )
    
    def compute_partials(self, inputs, outputs, partials):
        # Compute intermediate values
        propulsive_power = inputs["shaft_power_unit"] * inputs["n_lift_motors"]
        motor_eff = inputs["eta_motor"] * inputs["eta_cable"] * inputs["eta_electronics"]
        electrical_power = propulsive_power / motor_eff
        bat_power = electrical_power * inputs["epsilon_hover"]
        generator_power_unit = electrical_power * (1 - inputs["epsilon_hover"])
        
        # Partial derivatives for bat_energy
        partials["bat_energy", "shaft_power_unit"] = inputs["hover_time"] * inputs["epsilon_hover"] / (
            motor_eff)
        partials["bat_energy", "hover_time"] = bat_power
        partials["bat_energy", "epsilon_hover"] = electrical_power * inputs["hover_time"]
        partials["bat_energy", "eta_motor"] = -bat_power * inputs["hover_time"] / inputs["eta_motor"]
        partials["bat_energy", "eta_cable"] = -bat_power * inputs["hover_time"] / inputs["eta_cable"]
        partials["bat_energy", "eta_electronics"] = -bat_power * inputs["hover_time"] / inputs["eta_electronics"]
        partials["bat_energy", "eta_generator"] = 0.0
        partials["bat_energy", "eta_gasturbine"] = 0.0
        partials["bat_energy", "n_lift_motors"] = 0.0
        partials["bat_energy", "n_gens"] = 0.0
        
        # Partial derivatives for bat_power
        partials["bat_power", "prop_power"] = inputs["epsilon_hover"] / (
            motor_eff)
        partials["bat_power", "epsilon_hover"] = electrical_power
        partials["bat_power", "eta_motor"] = -bat_power / inputs["eta_motor"]
        partials["bat_power", "eta_cable"] = -bat_power / inputs["eta_cable"]
        partials["bat_power", "eta_electronics"] = -bat_power / inputs["eta_electronics"]
        partials["bat_power", "hover_time"] = 0.0
        partials["bat_power", "eta_generator"] = 0.0
        partials["bat_power", "eta_gasturbine"] = 0.0
        partials["bat_power", "n_lift_motors"] = 0.0
        partials["bat_power", "n_gens"] = 0.0
        
        # Partial derivatives for lift_motor_power_unit
        partials["lift_motor_power_unit", "prop_power"] = 1.0 / (
            inputs["eta_prop"] * inputs["n_lift_motors"])
        partials["lift_motor_power_unit", "eta_prop"] = -propulsive_power / (
            inputs["eta_prop"]**2 * inputs["n_lift_motors"])
        partials["lift_motor_power_unit", "n_lift_motors"] = -propulsive_power / inputs["n_lift_motors"]**2
        partials["lift_motor_power_unit", "epsilon_hover"] = 0.0
        partials["lift_motor_power_unit", "hover_time"] = 0.0
        partials["lift_motor_power_unit", "eta_motor"] = 0.0
        partials["lift_motor_power_unit", "eta_cable"] = 0.0
        partials["lift_motor_power_unit", "eta_electronics"] = 0.0
        partials["lift_motor_power_unit", "eta_generator"] = 0.0
        partials["lift_motor_power_unit", "eta_gasturbine"] = 0.0
        partials["lift_motor_power_unit", "n_gens"] = 0.0
        
        # Partial derivatives for generator_power_unit
        partials["generator_power_unit", "prop_power"] = (1 - inputs["epsilon_hover"]) / (
            motor_eff * inputs["n_gens"])
        partials["generator_power_unit", "eta_prop"] = -propulsive_power * (1 - inputs["epsilon_hover"]) / (
            motor_eff**2 * inputs["n_gens"])
        partials["generator_power_unit", "epsilon_hover"] = -electrical_power / inputs["n_gens"]
        partials["generator_power_unit", "eta_motor"] = -generator_power_unit / inputs["eta_motor"]
        partials["generator_power_unit", "eta_cable"] = -generator_power_unit / inputs["eta_cable"]
        partials["generator_power_unit", "eta_electronics"] = -generator_power_unit / inputs["eta_electronics"]
        partials["generator_power_unit", "n_gens"] = -generator_power_unit / inputs["n_gens"]
        partials["generator_power_unit", "hover_time"] = 0.0
        partials["generator_power_unit", "eta_generator"] = 0.0
        partials["generator_power_unit", "eta_gasturbine"] = 0.0
        partials["generator_power_unit", "n_lift_motors"] = 0.0
        
        # Partial derivatives for gasturbine_power
        gen_eff = inputs["eta_generator"] * inputs["eta_gasturbine"]
        partials["gasturbine_power_total", "prop_power"] = (1 - inputs["epsilon_hover"]) / (
            motor_eff * inputs["n_gens"] * gen_eff)
        partials["gasturbine_power", "eta_prop"] = -propulsive_power * (1 - inputs["epsilon_hover"]) / (
            motor_eff * inputs["n_gens"] * gen_eff)
        partials["gasturbine_power_total", "epsilon_hover"] = -electrical_power / (inputs["n_gens"] * gen_eff)
        partials["gasturbine_power_total", "eta_motor"] = -generator_power_unit / (inputs["eta_motor"] * gen_eff)
        partials["gasturbine_power_total", "eta_cable"] = -generator_power_unit / (inputs["eta_cable"] * gen_eff)
        partials["gasturbine_power_total", "eta_electronics"] = -generator_power_unit / (inputs["eta_electronics"] * gen_eff)
        partials["gasturbine_power_total", "n_gens"] = -generator_power_unit / (inputs["n_gens"] * gen_eff)
        partials["gasturbine_power_total", "eta_generator"] = -outputs["gasturbine_power_total"] / inputs["eta_generator"]
        partials["gasturbine_power_total", "eta_gasturbine"] = -outputs["gasturbine_power_total"] / inputs["eta_gasturbine"]
        partials["gasturbine_power_total", "hover_time"] = 0.0
        partials["gasturbine_power_total", "n_lift_motors"] = 0.0


class HoverEnergyGroup(om.Group):
    """Group containing hover energy analysis components."""
    
    def setup(self):

        # Add force computation
        self.add_subsystem("hover_force",
                          ComputeHoverForce(),
                          promotes_inputs=["mass_aircraft"],
                          promotes_outputs=["hover_force"])
        
        # Add propeller performance (from propeller.py)
        self.add_subsystem("propeller",
                          Propeller(),
                          promotes_inputs=["*"], 
                          promotes_outputs=["shaft_power_unit"])
        
        # Add power and energy computation
        self.add_subsystem("power_energy",
                          ComputeHoverPowerEnergy(),
                          promotes_inputs=["hover_time", "epsilon_hover" ,"n_lift_motors","shaft_power_unit",
                                         "n_gens", "eta_motor", "eta_cable",
                                         "eta_electronics", "eta_generator", "eta_gasturbine"],
                          promotes_outputs=["bat_energy", "bat_power",
                                          "lift_motor_power_unit", "generator_power_unit",
                                          "gasturbine_power_total"])
        
        # Connect hover force to propeller thrust_total
        self.connect("hover_force", "thrust_total")
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
    ivc.add_output("hover_time", val=120.0, units="s")
    ivc.add_output("epsilon_hover", val=0.4)
    ivc.add_output("n_lift_motors", val=8.0)
    ivc.add_output("n_gens", val=2.0)
    ivc.add_output("eta_motor", val=0.95)
    ivc.add_output("d_blade", val=3.0, units="m")
    ivc.add_output("d_hub", val=1.0, units="m")
    ivc.add_output("vel", val=0.0, units="m/s")
    ivc.add_output("eta_cable", val=0.98)
    ivc.add_output("eta_electronics", val=0.95)
    ivc.add_output("eta_generator", val=0.95)
    ivc.add_output("eta_prop", val=0.8)
    ivc.add_output("eta_gasturbine", val=0.30)
    ivc.add_output("rho", val=1.225, units="kg/m**3")
    ivc.add_output("eta_hover", val=0.5)
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.connect("n_lift_motors", "n_motors")
    model.add_subsystem('hover', HoverEnergyGroup(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Generate N2 diagram
    om.n2(prob)
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nHover Energy Results:')
    print('Hover Force Required:', prob.get_val('hover.hover_force')[0], 'N')
    print('Propulsive Power:', prob.get_val('hover.shaft_power_unit')[0]/1000, 'kW')
    print('bat Energy Required:', prob.get_val('hover.bat_energy')[0]/3.6e6, 'kWh')
    print('bat Power:', prob.get_val('hover.bat_power')[0]/1000, 'kW')
    print('Motor Power (each):', prob.get_val('hover.lift_motor_power_unit')[0]/1000, 'kW')
    print('Generator Power (each):', prob.get_val('hover.generator_power_unit')[0]/1000, 'kW')
    print('Gas Turbine Power (total):', prob.get_val('hover.gasturbine_power_total')[0]/1000, 'kW')
    
    # Check partials
    # prob.check_partials(compact_print=True) 