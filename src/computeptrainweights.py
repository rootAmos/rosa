from __future__ import annotations

import numpy as np
import openmdao.api as om


class PowertrainWeight(om.ExplicitComponent):
    """Computes weights of electrical powertrain components."""
    
    def setup(self):
        # Power and energy inputs
        self.add_input("electrical_power", val=1.0, units="W",
                      desc="Total electrical power required")
        self.add_input("battery_power", val=1.0, units="W",
                      desc="Maximum battery power required")
        self.add_input("battery_energy", val=1.0, units="J",
                      desc="Total battery energy required")
        self.add_input("motor_power", val=1.0, units="W",
                      desc="Power per motor")
        self.add_input("generator_power", val=1.0, units="W",
                      desc="Power per generator")
        self.add_input("gasturbine_power", val=1.0, units="W",
                      desc="Gas turbine power required")
        
        # Component counts
        self.add_input("n_motors", val=8.0, units=None,
                      desc="Number of lift motors")
        self.add_input("n_generators", val=2.0, units=None,
                      desc="Number of generators")
        
        # Power/energy densities
        self.add_input("motor_power_density", val=5000.0, units="W/kg",
                      desc="Motor specific power")
        self.add_input("power_electronics_power_density", val=10000.0, units="W/kg",
                      desc="Power electronics specific power")
        self.add_input("cable_power_density", val=20000.0, units="W/kg",
                      desc="Cable specific power")
        self.add_input("battery_power_density", val=3000.0, units="W/kg",
                      desc="Battery specific power")
        self.add_input("battery_energy_density", val=200.0, units="W*h/kg",
                      desc="Battery specific energy")
        self.add_input("turbine_power_density", val=2000.0, units="W/kg",
                      desc="Gas turbine specific power")
        self.add_input("generator_power_density", val=3000.0, units="W/kg",
                      desc="Generator specific power")
        
        # Add outputs
        self.add_output("w_motors", units="N",
                       desc="Motor weight")
        self.add_output("w_power_electronics", units="N",
                       desc="Power electronics weight")
        self.add_output("w_cables", units="N",
                       desc="Cable weight")
        self.add_output("w_battery", units="N",
                       desc="Battery weight")
        self.add_output("w_turbines", units="N",
                       desc="Gas turbine weight")
        self.add_output("w_generators", units="N",
                       desc="Generator weight")
        self.add_output("w_ptrain", units="N",
                       desc="Total electrical system weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):
        g = 9.81  # for converting mass to weight in N
        
        # Motor weight
        motor_mass = (inputs["motor_power"] * inputs["n_motors"] / 
                     inputs["motor_power_density"])
        outputs["w_motors"] = motor_mass * g
        
        # Power electronics weight
        pe_mass = inputs["electrical_power"] / inputs["power_electronics_power_density"]
        outputs["w_power_electronics"] = pe_mass * g
        
        # Cable weight
        cable_mass = inputs["electrical_power"] / inputs["cable_power_density"]
        outputs["w_cables"] = cable_mass * g
        
        # Battery weight (maximum of power and energy limits)
        battery_mass_power = inputs["battery_power"] / inputs["battery_power_density"]
        battery_mass_energy = (inputs["battery_energy"] / 3600) / inputs["battery_energy_density"]
        battery_mass = max(battery_mass_power, battery_mass_energy)
        outputs["w_battery"] = battery_mass * g
        
        # Gas turbine weight
        turbine_mass = inputs["gasturbine_power"] / inputs["turbine_power_density"]
        outputs["w_turbines"] = turbine_mass * g
        
        # Generator weight
        generator_mass = (inputs["generator_power"] * inputs["n_generators"] / 
                         inputs["generator_power_density"])
        outputs["w_generators"] = generator_mass * g
        
        # Total electrical system weight
        outputs["w_ptrain"] = (outputs["w_motors"] + outputs["w_power_electronics"] + 
                                 outputs["w_cables"] + outputs["w_battery"] + 
                                 outputs["w_turbines"] + outputs["w_generators"])

    def compute_partials(self, inputs, partials):
        g = 9.81
        
        # Motor weight partials
        partials["w_motors", "motor_power"] = inputs["n_motors"] * g / inputs["motor_power_density"]
        partials["w_motors", "n_motors"] = inputs["motor_power"] * g / inputs["motor_power_density"]
        partials["w_motors", "motor_power_density"] = -inputs["motor_power"] * inputs["n_motors"] * g / \
                                                    inputs["motor_power_density"]**2
        
        # Power electronics weight partials
        partials["w_power_electronics", "electrical_power"] = g / inputs["power_electronics_power_density"]
        partials["w_power_electronics", "power_electronics_power_density"] = \
            -inputs["electrical_power"] * g / inputs["power_electronics_power_density"]**2
        
        # Cable weight partials
        partials["w_cables", "electrical_power"] = g / inputs["cable_power_density"]
        partials["w_cables", "cable_power_density"] = \
            -inputs["electrical_power"] * g / inputs["cable_power_density"]**2
        
        # Battery weight partials
        battery_mass_power = inputs["battery_power"] / inputs["battery_power_density"]
        battery_mass_energy = (inputs["battery_energy"] / 3600) / inputs["battery_energy_density"]
        
        if battery_mass_power >= battery_mass_energy:
            partials["w_battery", "battery_power"] = g / inputs["battery_power_density"]
            partials["w_battery", "battery_power_density"] = \
                -inputs["battery_power"] * g / inputs["battery_power_density"]**2
            partials["w_battery", "battery_energy"] = 0.0
            partials["w_battery", "battery_energy_density"] = 0.0
        else:
            partials["w_battery", "battery_power"] = 0.0
            partials["w_battery", "battery_power_density"] = 0.0
            partials["w_battery", "battery_energy"] = g / (3600 * inputs["battery_energy_density"])
            partials["w_battery", "battery_energy_density"] = \
                -(inputs["battery_energy"] / 3600) * g / inputs["battery_energy_density"]**2
        
        # Gas turbine weight partials
        partials["w_turbines", "gasturbine_power"] = g / inputs["turbine_power_density"]
        partials["w_turbines", "turbine_power_density"] = \
            -inputs["gasturbine_power"] * g / inputs["turbine_power_density"]**2
        
        # Generator weight partials
        partials["w_generators", "generator_power"] = inputs["n_generators"] * g / inputs["generator_power_density"]
        partials["w_generators", "n_generators"] = inputs["generator_power"] * g / inputs["generator_power_density"]
        partials["w_generators", "generator_power_density"] = \
            -inputs["generator_power"] * inputs["n_generators"] * g / inputs["generator_power_density"]**2
        
        # Total electrical weight partials
        for output in ["w_motors", "w_power_electronics", "w_cables", "w_battery", "w_turbines", "w_generators"]:
            for input_name in inputs:
                if (output, input_name) in partials:
                    partials["w_ptrain", input_name] = partials[output, input_name]
                else:
                    partials["w_ptrain", input_name] = 0.0


if __name__ == "__main__":
    # Create test problem
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output("electrical_power", val=1000000.0, units="W")
    ivc.add_output("battery_power", val=400000.0, units="W")
    ivc.add_output("battery_energy", val=1000000000.0, units="J")
    ivc.add_output("motor_power", val=125000.0, units="W")
    ivc.add_output("generator_power", val=300000.0, units="W")
    ivc.add_output("gasturbine_power", val=600000.0, units="W")
    ivc.add_output("n_motors", val=8.0)
    ivc.add_output("n_generators", val=2.0)
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('ptrain_weight', PowertrainWeight(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nPowertrain Weight Results:')
    print('Motors Weight:', prob.get_val('ptrain_weight.w_motors')[0], 'N')
    print('Power Electronics Weight:', prob.get_val('ptrain_weight.w_power_electronics')[0], 'N')
    print('Cables Weight:', prob.get_val('ptrain_weight.w_cables')[0], 'N')
    print('Battery Weight:', prob.get_val('ptrain_weight.w_battery')[0], 'N')
    print('Gas Turbines Weight:', prob.get_val('ptrain_weight.w_turbines')[0], 'N')
    print('Generators Weight:', prob.get_val('ptrain_weight.w_generators')[0], 'N')
    print('Total Powertrain Weight:', prob.get_val('ptrain_weight.w_ptrain')[0], 'N')
    
    # Check partials
    # prob.check_partials(compact_print=True) 