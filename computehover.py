from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any
from computepropeller import PropellerPerformance


class HoverForce(om.ExplicitComponent):
    """Computes force required for hover."""
    
    def setup(self):
        self.add_input("mass_aircraft", val=1.0, units="kg",
                      desc="Total aircraft mass")
        
        self.add_output("hover_force", units="N",
                       desc="Force required for hover")
        
        self.declare_partials("hover_force", "mass_aircraft", method="exact")
        
    def compute(self, inputs, outputs):
        g = 9.81
        outputs["hover_force"] = inputs["mass_aircraft"] * g
        
    def compute_partials(self, inputs, partials):
        g = 9.81
        partials["hover_force", "mass_aircraft"] = g


class HoverPowerEnergy(om.ExplicitComponent):
    """Computes power and energy requirements from propulsive power."""
    
    def setup(self):
        self.add_input("prop_power", val=1.0, units="W",
                      desc="Propulsive power required")
        self.add_input("eta_prop", val=0.8, units=None,
                      desc="Propeller efficiency")
        self.add_input("hover_time", val=60.0, units="s",
                      desc="Time spent in hover")
        self.add_input("epsilon_hover", val=0.3, units=None,
                      desc="Hover power hybridization ratio (battery fraction)")
        self.add_input("n_lift_motors", val=8.0, units=None,
                      desc="Number of lift motors")
        self.add_input("n_generators", val=2.0, units=None,
                      desc="Number of generators")
        
        # Efficiencies
        self.add_input("eta_motor", val=0.95, units=None,
                      desc="Motor efficiency")
        self.add_input("eta_cable", val=0.98, units=None,
                      desc="Cable efficiency")
        self.add_input("eta_electronics", val=0.95, units=None,
                      desc="Power electronics efficiency")
        self.add_input("eta_generator", val=0.95, units=None,
                      desc="Generator efficiency")
        self.add_input("eta_gasturbine", val=0.30, units=None,
                      desc="Gas turbine thermal efficiency")
        
        # Add outputs
        self.add_output("battery_energy", units="J",
                       desc="Total battery energy required")
        self.add_output("battery_power", units="W",
                       desc="Maximum battery power required")
        self.add_output("motor_power", units="W",
                       desc="Maximum power per lift motor")
        self.add_output("generator_power", units="W",
                       desc="Maximum power per generator")
        self.add_output("gasturbine_power", units="W",
                       desc="Gas turbine power required")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):
        # Compute shaft power required (accounting for propeller efficiency)
        shaft_power = inputs["prop_power"] / inputs["eta_prop"]
        
        # Compute electrical power required (accounting for motor efficiencies)
        electrical_power = shaft_power / (
            inputs["eta_motor"] * inputs["eta_cable"] * inputs["eta_electronics"]
        )
        
        # Split power between battery and generator based on hybridization ratio
        battery_power = electrical_power * inputs["epsilon_hover"]
        generator_power = electrical_power * (1 - inputs["epsilon_hover"])
        
        # Compute battery energy required
        outputs["battery_energy"] = battery_power * inputs["hover_time"]
        
        # Compute maximum power requirements
        outputs["battery_power"] = battery_power
        outputs["motor_power"] = shaft_power / inputs["n_lift_motors"]
        outputs["generator_power"] = generator_power / inputs["n_generators"]
        outputs["gasturbine_power"] = outputs["generator_power"] / (
            inputs["eta_generator"] * inputs["eta_gasturbine"]
        )
    
    def compute_partials(self, inputs, outputs, partials):
        # Compute intermediate values
        shaft_power = inputs["prop_power"] / inputs["eta_prop"]
        motor_eff = inputs["eta_motor"] * inputs["eta_cable"] * inputs["eta_electronics"]
        electrical_power = shaft_power / motor_eff
        battery_power = electrical_power * inputs["epsilon_hover"]
        generator_power = electrical_power * (1 - inputs["epsilon_hover"])
        
        # Partial derivatives for battery_energy
        partials["battery_energy", "prop_power"] = inputs["hover_time"] * inputs["epsilon_hover"] / (
            inputs["eta_prop"] * motor_eff)
        partials["battery_energy", "eta_prop"] = -inputs["prop_power"] * inputs["hover_time"] * \
            inputs["epsilon_hover"] / (inputs["eta_prop"]**2 * motor_eff)
        partials["battery_energy", "hover_time"] = battery_power
        partials["battery_energy", "epsilon_hover"] = electrical_power * inputs["hover_time"]
        partials["battery_energy", "eta_motor"] = -battery_power * inputs["hover_time"] / inputs["eta_motor"]
        partials["battery_energy", "eta_cable"] = -battery_power * inputs["hover_time"] / inputs["eta_cable"]
        partials["battery_energy", "eta_electronics"] = -battery_power * inputs["hover_time"] / inputs["eta_electronics"]
        partials["battery_energy", "eta_generator"] = 0.0
        partials["battery_energy", "eta_gasturbine"] = 0.0
        partials["battery_energy", "n_lift_motors"] = 0.0
        partials["battery_energy", "n_generators"] = 0.0
        
        # Partial derivatives for battery_power
        partials["battery_power", "prop_power"] = inputs["epsilon_hover"] / (
            inputs["eta_prop"] * motor_eff)
        partials["battery_power", "eta_prop"] = -inputs["prop_power"] * inputs["epsilon_hover"] / (
            inputs["eta_prop"]**2 * motor_eff)
        partials["battery_power", "epsilon_hover"] = electrical_power
        partials["battery_power", "eta_motor"] = -battery_power / inputs["eta_motor"]
        partials["battery_power", "eta_cable"] = -battery_power / inputs["eta_cable"]
        partials["battery_power", "eta_electronics"] = -battery_power / inputs["eta_electronics"]
        partials["battery_power", "hover_time"] = 0.0
        partials["battery_power", "eta_generator"] = 0.0
        partials["battery_power", "eta_gasturbine"] = 0.0
        partials["battery_power", "n_lift_motors"] = 0.0
        partials["battery_power", "n_generators"] = 0.0
        
        # Partial derivatives for motor_power
        partials["motor_power", "prop_power"] = 1.0 / (
            inputs["eta_prop"] * inputs["n_lift_motors"])
        partials["motor_power", "eta_prop"] = -inputs["prop_power"] / (
            inputs["eta_prop"]**2 * inputs["n_lift_motors"])
        partials["motor_power", "n_lift_motors"] = -shaft_power / inputs["n_lift_motors"]**2
        partials["motor_power", "epsilon_hover"] = 0.0
        partials["motor_power", "hover_time"] = 0.0
        partials["motor_power", "eta_motor"] = 0.0
        partials["motor_power", "eta_cable"] = 0.0
        partials["motor_power", "eta_electronics"] = 0.0
        partials["motor_power", "eta_generator"] = 0.0
        partials["motor_power", "eta_gasturbine"] = 0.0
        partials["motor_power", "n_generators"] = 0.0
        
        # Partial derivatives for generator_power
        partials["generator_power", "prop_power"] = (1 - inputs["epsilon_hover"]) / (
            inputs["eta_prop"] * motor_eff * inputs["n_generators"])
        partials["generator_power", "eta_prop"] = -inputs["prop_power"] * (1 - inputs["epsilon_hover"]) / (
            inputs["eta_prop"]**2 * motor_eff * inputs["n_generators"])
        partials["generator_power", "epsilon_hover"] = -electrical_power / inputs["n_generators"]
        partials["generator_power", "eta_motor"] = -generator_power / inputs["eta_motor"]
        partials["generator_power", "eta_cable"] = -generator_power / inputs["eta_cable"]
        partials["generator_power", "eta_electronics"] = -generator_power / inputs["eta_electronics"]
        partials["generator_power", "n_generators"] = -generator_power / inputs["n_generators"]
        partials["generator_power", "hover_time"] = 0.0
        partials["generator_power", "eta_generator"] = 0.0
        partials["generator_power", "eta_gasturbine"] = 0.0
        partials["generator_power", "n_lift_motors"] = 0.0
        
        # Partial derivatives for gasturbine_power
        gen_eff = inputs["eta_generator"] * inputs["eta_gasturbine"]
        partials["gasturbine_power", "prop_power"] = (1 - inputs["epsilon_hover"]) / (
            inputs["eta_prop"] * motor_eff * inputs["n_generators"] * gen_eff)
        partials["gasturbine_power", "eta_prop"] = -inputs["prop_power"] * (1 - inputs["epsilon_hover"]) / (
            inputs["eta_prop"]**2 * motor_eff * inputs["n_generators"] * gen_eff)
        partials["gasturbine_power", "epsilon_hover"] = -electrical_power / (inputs["n_generators"] * gen_eff)
        partials["gasturbine_power", "eta_motor"] = -generator_power / (inputs["eta_motor"] * gen_eff)
        partials["gasturbine_power", "eta_cable"] = -generator_power / (inputs["eta_cable"] * gen_eff)
        partials["gasturbine_power", "eta_electronics"] = -generator_power / (inputs["eta_electronics"] * gen_eff)
        partials["gasturbine_power", "n_generators"] = -generator_power / (inputs["n_generators"] * gen_eff)
        partials["gasturbine_power", "eta_generator"] = -outputs["gasturbine_power"] / inputs["eta_generator"]
        partials["gasturbine_power", "eta_gasturbine"] = -outputs["gasturbine_power"] / inputs["eta_gasturbine"]
        partials["gasturbine_power", "hover_time"] = 0.0
        partials["gasturbine_power", "n_lift_motors"] = 0.0


class HoverEnergyGroup(om.Group):
    """Group containing hover energy analysis components."""
    
    def setup(self):
        # Add force computation
        self.add_subsystem("hover_force",
                          HoverForce(),
                          promotes_inputs=["mass_aircraft"],
                          promotes_outputs=["hover_force"])
        
        # Add propeller performance (from propeller.py)
        self.add_subsystem("propeller",
                          PropellerPerformance(),
                          promotes_inputs=["rho"])
        
        # Add power and energy computation
        self.add_subsystem("power_energy",
                          HoverPowerEnergy(),
                          promotes_inputs=["hover_time", "epsilon_hover", "n_lift_motors",
                                         "n_generators", "eta_motor", "eta_cable",
                                         "eta_electronics", "eta_generator", "eta_gasturbine"],
                          promotes_outputs=["battery_energy", "battery_power",
                                          "motor_power", "generator_power",
                                          "gasturbine_power"])
        
        # Connect hover force to propeller thrust
        self.connect("hover_force", "propeller.thrust")
        
        # Set velocity to zero for hover condition
        self.set_input_defaults("propeller.velocity", val=0.0, units="m/s")
        
        # Connect propeller outputs to power_energy
        self.connect("propeller.power", "power_energy.prop_power")
        self.connect("propeller.efficiency", "power_energy.eta_prop")


if __name__ == "__main__":
    # Create problem
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output("mass_aircraft", val=2000.0, units="kg")
    ivc.add_output("hover_time", val=120.0, units="s")
    ivc.add_output("epsilon_hover", val=0.4)
    ivc.add_output("n_lift_motors", val=8.0)
    ivc.add_output("n_generators", val=2.0)
    ivc.add_output("eta_motor", val=0.95)
    ivc.add_output("eta_cable", val=0.98)
    ivc.add_output("eta_electronics", val=0.95)
    ivc.add_output("eta_generator", val=0.95)
    ivc.add_output("eta_gasturbine", val=0.30)
    ivc.add_output("rho", val=1.225, units="kg/m**3")
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('hover', HoverEnergyGroup(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nHover Energy Results:')
    print('Hover Force Required:', prob.get_val('hover.hover_force')[0], 'N')
    print('Propulsive Power:', prob.get_val('hover.propeller.power')[0]/1000, 'kW')
    print('Propeller Efficiency:', prob.get_val('hover.propeller.efficiency')[0])
    print('Battery Energy Required:', prob.get_val('hover.battery_energy')[0]/3.6e6, 'kWh')
    print('Battery Power:', prob.get_val('hover.battery_power')[0]/1000, 'kW')
    print('Motor Power (each):', prob.get_val('hover.motor_power')[0]/1000, 'kW')
    print('Generator Power (each):', prob.get_val('hover.generator_power')[0]/1000, 'kW')
    print('Gas Turbine Power:', prob.get_val('hover.gasturbine_power')[0]/1000, 'kW')
    
    # Check partials
    prob.check_partials(compact_print=True) 