from __future__ import annotations

import numpy as np
import openmdao.api as om


class FuelSystemWeight(om.ExplicitComponent):
    """Computes fuel system weight using Raymer's equation.
    
    Gudmunssonn Equation 6-85
    """
    
    def setup(self):
        self.add_input("v_fuel_tot", val=100.0, units="galUS",
                      desc="total fuel quantity")
        self.add_input("v_fuel_int", val=80.0, units="galUS",
                      desc="integral tank fuel quantity")
        self.add_input("n_tank", val=2.0, units=None,
                      desc="Number of fuel tanks")
        self.add_input("n_turbine", val=2.0, units=None,
                      desc="Number of turbines")
        
        self.add_output("w_fuel_system", units="lbf",
                       desc="Fuel system weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):
        v_fuel_tot = inputs["v_fuel_tot"]
        v_fuel_int = inputs["v_fuel_int"]
        n_tank = inputs["n_tank"]
        n_turbine = inputs["n_turbine"]
        
        outputs["w_fuel_system"] = (
            2.49 
            * v_fuel_tot**0.726 
            * (v_fuel_tot / (v_fuel_tot + v_fuel_int))**0.363 
            * n_tank**0.242 
            * n_turbine**0.157
        )
        


class PropellerWeight(om.ExplicitComponent):
    """Computes propeller weight.

    Gundlach Equation 6-54
    """
    
    def setup(self):

        self.add_input("k_prop", val=1.0, units=None,
                      desc="Propeller weight scaling coefficient")
        self.add_input("n_props", val=1.0, units=None,
                      desc="Number of propellers")
        self.add_input("n_blades", val=1.0, units=None,
                      desc="Number of blades")
        self.add_input("d_blades", val=1.0, units="m",
                      desc="Propeller diameter")
        self.add_input("p_shaft", val=1.0, units="hp",
                      desc="Shaft power per propeller")
        
        self.add_output("w_prop", units="lbf",
                       desc="Propeller weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):

        n_props = inputs["n_props"]
        n_blades = inputs["n_blades"]
        p_shaft = inputs["p_shaft"]
        d_blades = inputs["d_blades"]
        k_prop = inputs["k_prop"]
        
        
        outputs["w_prop"] = k_prop * n_props * n_blades ** 0.391 * ( (d_blades * p_shaft) / (n_props * 1000) ) ** 0.782


class TurbineWeight(om.ExplicitComponent):
    """Computes gas turbine weight using empirical equation.
    
    Gudmunssonn Equation 6-107
    """
    
    def setup(self):
        self.add_input("p_turbine", val=1000.0, units="hp",
                      desc="Maximum engine power")
        
        self.add_output("w_turbine", units="lbf",
                       desc="turbine engine weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        p_turbine = inputs["p_turbine"]
        outputs["w_turbine"] = 71.65 + 0.3658 * p_turbine

"""
May not be needed. Double count with NacelleWeight
class InstalledTurbineWeight(om.ExplicitComponent):
    
    # Computes installed turbine weight using Raymer's equation
    
    
    def setup(self):
        self.add_input("w_turbine", val=500.0, units="lbf",
                      desc="Weight of single turbine")
        self.add_input("n_turbine", val=2.0, units=None,
                      desc="Number of turbines")
        
        self.add_output("w_turbines", units="lbf",
                       desc="Total installed turbine engines weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        w_turbine = inputs["w_turbine"]
        n_turbine = inputs["n_turbine"]
        
        outputs["w_turbines"] = 2.575 * w_turbine**0.922 * n_turbine 

"""

class NacelleWeight(om.ExplicitComponent):
    """Computes nacelle weight based on Raymer's equations.
    
      Gudmunssonn Equation 6-73
    """
    
    def setup(self):
        self.add_input("n_motors", val=1.0, units=None,
                      desc="Number of motors")
        self.add_input("n_gens", val=1.0, units=None,
                      desc="Number of gens")
        self.add_input("n_turbines", val=1.0, units=None,
                      desc="Number of turbines")
        self.add_input("p_motor", val=1.0, units="W",
                      desc="Power per motor")
        self.add_input("p_gen", val=1.0, units="W",
                      desc="Power per generator")
        self.add_input("p_turbine", val=1.0, units="W",
                      desc="Gas turbine power required")

        self.add_output("w_nacelles", units="N",
                       desc="Nacelles weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):

        
        outputs["w_nacelles"] = 0.14 * ( inputs["p_motor"] * inputs["n_motors"] + inputs["p_gen"] *\
                                         inputs["n_gens"]  + inputs["p_turbine"] * inputs["n_turbines"] ) 

class ElecPowertrainWeight(om.ExplicitComponent):
    """Computes weights of electrical powertrain components."""
    
    def setup(self):
        # Power and energy inputs
        self.add_input("p_elec", val=1.0, units="W",
                      desc="Total electrical power required")
        self.add_input("p_bat", val=1.0, units="W",
                      desc="Maximum bat power required")
        self.add_input("e_bat", val=1.0, units="J",
                      desc="Total bat energy required")
        self.add_input("p_motor", val=1.0, units="W",
                      desc="Power per motor")
        self.add_input("p_gen", val=1.0, units="W",
                      desc="Power per generator")
        self.add_input("p_turbine", val=1.0, units="W",
                      desc="Gas turbine power required")
        
        # Component counts
        self.add_input("n_motors", val=8.0, units=None,
                      desc="Number of lift motors")
        self.add_input("n_gens", val=2.0, units=None,
                      desc="Number of gens")
        
        # Power/energy densities
        self.add_input("sp_motor", val=5000.0, units="W/kg",
                      desc="Motor specific power")
        self.add_input("sp_pe", val=10000.0, units="W/kg",
                      desc="Power electronics specific power")
        self.add_input("sp_cbl", val=20000.0, units="W/kg",
                      desc="Cable specific power")
        self.add_input("sp_bat", val=3000.0, units="W/kg",
                      desc="bat specific power")
        self.add_input("se_bat", val=200.0, units="W*h/kg",
                      desc="bat specific energy")
        self.add_input("turbine_power_density", val=2000.0, units="W/kg",
                      desc="Gas turbine specific power")
        self.add_input("sp_gen", val=3000.0, units="W/kg",
                      desc="Generator specific power")
        
        # Add outputs
        self.add_output("w_motors", units="N",
                       desc="Motor weight")
        self.add_output("w_pe", units="N",
                       desc="Power electronics weight")
        self.add_output("w_cbls", units="N",
                       desc="Cable weight")
        self.add_output("w_bat", units="N",
                       desc="bat weight")
        self.add_output("w_turbines", units="N",
                       desc="Gas turbine weight")
        self.add_output("w_gens", units="N",
                       desc="Generator weight")
        self.add_output("w_elec_ptrain", units="N",
                       desc="Total electrical system weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):
        g = 9.81  # for converting mass to weight in N
        
        # Motor weight
        motor_mass = (inputs["p_motor"] * inputs["n_motors"] / 
                     inputs["sp_motor"])
        outputs["w_motors"] = motor_mass * g
        
        # Power electronics weight
        pe_mass = inputs["p_elec"] / inputs["sp_pe"]
        outputs["w_pe"] = pe_mass * g
        
        # Cable weight
        cable_mass = inputs["p_elec"] / inputs["sp_cbl"]
        outputs["w_cbls"] = cable_mass * g
        
        # bat weight (maximum of power and energy limits)
        bat_mass_power = inputs["p_bat"] / inputs["sp_bat"]
        bat_mass_energy = (inputs["e_bat"] / 3600) / inputs["se_bat"]
        bat_mass = max(bat_mass_power, bat_mass_energy)
        outputs["w_bat"] = bat_mass * g
        
        # Gas turbine weight
        turbine_mass = inputs["p_turbine"] / inputs["turbine_power_density"]
        outputs["w_turbines"] = turbine_mass * g
        
        # Generator weight
        generator_mass = (inputs["p_gen"] * inputs["n_gens"] / 
                         inputs["sp_gen"])
        outputs["w_gens"] = generator_mass * g
        
        # Total electrical system weight
        outputs["w_elec_ptrain"] = (outputs["w_motors"] + outputs["w_pe"] + 
                                 outputs["w_cbls"] + outputs["w_bat"] + 
                                 outputs["w_turbines"] + outputs["w_gens"])

    def compute_partials(self, inputs, partials):
        g = 9.81
        
        # Motor weight partials
        partials["w_motors", "p_motor"] = inputs["n_motors"] * g / inputs["sp_motor"]
        partials["w_motors", "n_motors"] = inputs["p_motor"] * g / inputs["sp_motor"]
        partials["w_motors", "sp_motor"] = -inputs["p_motor"] * inputs["n_motors"] * g / \
                                                    inputs["sp_motor"]**2
        
        # Power electronics weight partials
        partials["w_pe", "p_elec"] = g / inputs["sp_pe"]
        partials["w_pe", "sp_pe"] = \
            -inputs["p_elec"] * g / inputs["sp_pe"]**2
        
        # Cable weight partials
        partials["w_cbls", "p_elec"] = g / inputs["sp_cbl"]
        partials["w_cbls", "sp_cbl"] = \
            -inputs["p_elec"] * g / inputs["sp_cbl"]**2
        
        # bat weight partials
        bat_mass_power = inputs["p_bat"] / inputs["sp_bat"]
        bat_mass_energy = (inputs["e_bat"] / 3600) / inputs["se_bat"]
        
        if bat_mass_power >= bat_mass_energy:
            partials["w_bat", "p_bat"] = g / inputs["sp_bat"]
            partials["w_bat", "sp_bat"] = \
                -inputs["p_bat"] * g / inputs["sp_bat"]**2
            partials["w_bat", "e_bat"] = 0.0
            partials["w_bat", "se_bat"] = 0.0
        else:
            partials["w_bat", "p_bat"] = 0.0
            partials["w_bat", "sp_bat"] = 0.0
            partials["w_bat", "e_bat"] = g / (3600 * inputs["se_bat"])
            partials["w_bat", "se_bat"] = \
                -(inputs["e_bat"] / 3600) * g / inputs["se_bat"]**2
        
        # Gas turbine weight partials
        partials["w_turbines", "p_turbine"] = g / inputs["turbine_power_density"]
        partials["w_turbines", "turbine_power_density"] = \
            -inputs["p_turbine"] * g / inputs["turbine_power_density"]**2
        
        # Generator weight partials
        partials["w_gens", "p_gen"] = inputs["n_gens"] * g / inputs["sp_gen"]
        partials["w_gens", "n_gens"] = inputs["p_gen"] * g / inputs["sp_gen"]
        partials["w_gens", "sp_gen"] = \
            -inputs["p_gen"] * inputs["n_gens"] * g / inputs["sp_gen"]**2
        
        # Total electrical weight partials
        for output in ["w_motors", "w_pe", "w_cbls", "w_bat", "w_turbines", "w_gens"]:
            for input_name in inputs:
                if (output, input_name) in partials:
                    partials["w_elec_ptrain", input_name] = partials[output, input_name]
                else:
                    partials["w_elec_ptrain", input_name] = 0.0

class HybridPowertrainGroup(om.Group):
    """Group that combines all hybrid-electric powertrain components for weight calculation."""
    
    def setup(self):
        # Add all the individual components
        self.add_subsystem("fuel_system", FuelSystemWeight(), 
                          promotes_inputs=["Q_tot", "Q_int", "n_tank", "n_turbines"])
        
        self.add_subsystem("turbine_engine", TurbineWeight(),
                          promotes_inputs=["p_turbine"])
        
        
        self.add_subsystem("elec_powertrain", ElecPowertrainWeight(),
                          promotes_inputs=["p_elec", "p_bat", "e_bat", "p_motor", 
                                         "p_gen", "p_turbine", "n_motors", "n_gens",
                                         "sp_motor", "sp_pe", "sp_cbl", "sp_bat", 
                                         "se_bat", "turbine_power_density", "sp_gen"])
        
        self.add_subsystem("nacelle", NacelleWeight(),
                          promotes_inputs=["n_motors", "n_gens", "n_turbines",
                                         "p_motor", "p_gen", "p_turbine"])
        
        self.add_subsystem("propeller", PropellerWeight(),
                          promotes_inputs=["n_props", "d_blades", "p_shaft"])
        
        # Connect internal variables
        self.connect("turbine_engine.w_turbine", "nacelle.w_turbines")
        
        # Add total weight output
        self.add_subsystem("total_weight", om.AddSubtractComp(
            output_name="w_hybrid_ptrain",
            input_names=["w_fuel_system", "w_turbines", "w_elec_ptrain", 
                        "w_nacelles", "w_prop"],
            units="N",
            desc="Total hybrid powertrain weight"),
            promotes_outputs=["w_hybrid_ptrain"])
        
        # Connect component weights to total
        self.connect("fuel_system.w_fuel_system", "total_weight.w_fuel_system")
        self.connect("installed_turbine.w_turbines", "total_weight.w_turbines")
        self.connect("elec_powertrain.w_elec_ptrain", "total_weight.w_elec_ptrain")
        self.connect("nacelle.w_nacelles", "total_weight.w_nacelles")
        self.connect("propeller.w_prop", "total_weight.w_prop")

if __name__ == "__main__":
    # Create test problem
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    
    # Electrical system inputs
    ivc.add_output("p_elec", val=1000000.0, units="W")
    ivc.add_output("p_bat", val=400000.0, units="W")
    ivc.add_output("e_bat", val=1000000000.0, units="J")
    ivc.add_output("p_motor", val=125000.0, units="W")
    ivc.add_output("p_gen", val=300000.0, units="W")
    ivc.add_output("p_turbine", val=600000.0, units="W")
    ivc.add_output("n_motors", val=8.0)
    ivc.add_output("n_gens", val=2.0)
    
    # Power/energy densities
    ivc.add_output("sp_motor", val=5000.0, units="W/kg")
    ivc.add_output("sp_pe", val=10000.0, units="W/kg")
    ivc.add_output("sp_cbl", val=20000.0, units="W/kg")
    ivc.add_output("sp_bat", val=3000.0, units="W/kg")
    ivc.add_output("se_bat", val=200.0, units="W*h/kg")
    ivc.add_output("turbine_power_density", val=2000.0, units="W/kg")
    ivc.add_output("sp_gen", val=3000.0, units="W/kg")
    
    # Fuel system inputs
    ivc.add_output("Q_tot", val=500.0, units="galUS")
    ivc.add_output("Q_int", val=400.0, units="galUS")
    ivc.add_output("n_tank", val=2.0)
    ivc.add_output("n_turbine", val=2.0)
    
    # Turbine inputs
    ivc.add_output("p_max", val=1000.0, units="hp")
    
    # Propeller inputs
    ivc.add_output("n_props", val=2.0)
    ivc.add_output("d_blades", val=2.5, units="m")
    ivc.add_output("p_to", val=250000.0, units="W")
    
    # Nacelle additional inputs
    ivc.add_output("n_gens", val=2.0)
    ivc.add_output("n_turbines", val=2.0)
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('hybrid_ptrain', HybridPowertrainGroup(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nHybrid Powertrain Weight Results:')
    print('Fuel System Weight:', prob.get_val('hybrid_ptrain.fuel_system.w_fuel_system')[0], 'lbf')
    print('Turbine Engine Weight:', prob.get_val('hybrid_ptrain.turbine_engine.w_turbine')[0], 'lbf')
    print('Installed Turbines Weight:', prob.get_val('hybrid_ptrain.installed_turbine.w_turbines')[0], 'lbf')
    print('Electric Powertrain Weight:', prob.get_val('hybrid_ptrain.elec_powertrain.w_elec_ptrain')[0], 'N')
    print('Nacelles Weight:', prob.get_val('hybrid_ptrain.nacelle.w_nacelles')[0], 'N')
    print('Propeller Weight:', prob.get_val('hybrid_ptrain.propeller.w_prop')[0], 'N')
    print('Total Hybrid Powertrain Weight:', prob.get_val('w_hybrid_ptrain')[0], 'N')
    
    # Check partials (uncomment to verify derivatives)
    # prob.check_partials(compact_print=True) 