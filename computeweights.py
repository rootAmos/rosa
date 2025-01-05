from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any


class WingWeight(om.ExplicitComponent):
    """Computes wing weight based on Raymer's equations."""
    
    def setup(self):
        self.add_input("w_dg", val=1.0, units="N",
                      desc="Design gross weight")
        self.add_input("n_ult", val=1.0, units=None,
                      desc="Ultimate load factor")
        self.add_input("s_w", val=1.0, units="m**2",
                      desc="Wing area")
        self.add_input("ar_w", val=1.0, units=None,
                      desc="Wing aspect ratio")
        self.add_input("t_c_w", val=1.0, units=None,
                      desc="Wing thickness-to-chord ratio")
        self.add_input("sweep_w", val=1.0, units="rad",
                      desc="Wing sweep angle")
        
        self.add_output("w_wing", units="N",
                       desc="Wing weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        w_dg = inputs["w_dg"] / 9.81  # Convert to kg
        n_ult = inputs["n_ult"]
        s_w = inputs["s_w"]
        ar_w = inputs["ar_w"]
        t_c_w = inputs["t_c_w"]
        sweep_w = inputs["sweep_w"]
        
        outputs["w_wing"] = 0.036 * (w_dg ** 0.758) * (n_ult ** 0.6) * (s_w ** 0.5) * \
                           (ar_w ** 0.6) * (t_c_w ** -0.3) * \
                           ((np.cos(sweep_w)) ** -0.3) * 9.81


class FuselageWeight(om.ExplicitComponent):
    """Computes fuselage weight based on Raymer's equations."""
    
    def setup(self):
        self.add_input("w_dg", val=1.0, units="N",
                      desc="Design gross weight")
        self.add_input("n_ult", val=1.0, units=None,
                      desc="Ultimate load factor")
        self.add_input("l_f", val=1.0, units="m",
                      desc="Fuselage length")
        self.add_input("d_f", val=1.0, units="m",
                      desc="Fuselage diameter")
        self.add_input("v_p", val=1.0, units="m/s",
                      desc="Maximum velocity")
        
        self.add_output("w_fuselage", units="N",
                       desc="Fuselage weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        w_dg = inputs["w_dg"] / 9.81
        n_ult = inputs["n_ult"]
        l_f = inputs["l_f"]
        d_f = inputs["d_f"]
        v_p = inputs["v_p"]
        
        outputs["w_fuselage"] = 0.052 * (w_dg ** 0.177) * (n_ult ** 0.5) * \
                               (l_f ** 0.874) * (d_f ** 0.383) * \
                               (v_p ** 0.484) * 9.81


class LandingGearWeight(om.ExplicitComponent):
    """Computes landing gear weight based on Raymer's equations."""
    
    def setup(self):
        self.add_input("w_dg", val=1.0, units="N",
                      desc="Design gross weight")
        self.add_input("n_l", val=1.0, units=None,
                      desc="Ultimate landing load factor")
        self.add_input("l_m", val=1.0, units="m",
                      desc="Length of main landing gear")
        
        self.add_output("w_lg", units="N",
                       desc="Landing gear weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        w_dg = inputs["w_dg"] / 9.81
        n_l = inputs["n_l"]
        l_m = inputs["l_m"]
        
        outputs["w_lg"] = 0.095 * (w_dg ** 0.768) * (n_l ** 0.409) * \
                         (l_m ** 0.252) * 9.81


class NacelleWeight(om.ExplicitComponent):
    """Computes nacelle weight based on Raymer's equations."""
    
    def setup(self):
        self.add_input("w_engine", val=1.0, units="N",
                      desc="Engine weight")
        self.add_input("n_en", val=1.0, units=None,
                      desc="Number of engines")
        
        self.add_output("w_nacelle", units="N",
                       desc="Nacelle weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        w_engine = inputs["w_engine"] / 9.81
        n_en = inputs["n_en"]
        
        outputs["w_nacelle"] = 0.6 * w_engine * n_en * 9.81


class EngineWeight(om.ExplicitComponent):
    """Computes engine weight based on power."""
    
    def setup(self):
        self.add_input("p_to", val=1.0, units="W",
                      desc="Takeoff power")
        
        self.add_output("w_engine", units="N",
                       desc="Engine weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        p_to = inputs["p_to"] / 1000  # Convert to kW
        outputs["w_engine"] = 0.686 * (p_to ** 0.93) * 9.81


class PropellerWeight(om.ExplicitComponent):
    """Computes propeller weight."""
    
    def setup(self):
        self.add_input("n_p", val=1.0, units=None,
                      desc="Number of propellers")
        self.add_input("d_p", val=1.0, units="m",
                      desc="Propeller diameter")
        self.add_input("p_to", val=1.0, units="W",
                      desc="Takeoff power per propeller")
        
        self.add_output("w_prop", units="N",
                       desc="Propeller weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        n_p = inputs["n_p"]
        d_p = inputs["d_p"]
        p_to = inputs["p_to"] / 1000  # Convert to kW
        
        outputs["w_prop"] = 0.108 * n_p * (d_p ** 2.5) * (p_to ** 0.37) * 9.81


class SystemsWeight(om.ExplicitComponent):
    """Computes systems weight based on Raymer's equations."""
    
    def setup(self):
        self.add_input("w_dg", val=1.0, units="N",
                      desc="Design gross weight")
        
        self.add_output("w_systems", units="N",
                       desc="Systems weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        w_dg = inputs["w_dg"] / 9.81
        outputs["w_systems"] = 0.1 * w_dg * 9.81


class AircraftWeight(om.ExplicitComponent):
    """Computes total aircraft weight by summing component weights."""
    
    def setup(self):
        self.add_input("w_wing", val=1.0, units="N",
                      desc="Wing weight")
        self.add_input("w_fuselage", val=1.0, units="N",
                      desc="Fuselage weight")
        self.add_input("w_lg", val=1.0, units="N",
                      desc="Landing gear weight")
        self.add_input("w_nacelle", val=1.0, units="N",
                      desc="Nacelle weight")
        self.add_input("w_engine", val=1.0, units="N",
                      desc="Engine weight")
        self.add_input("w_prop", val=1.0, units="N",
                      desc="Propeller weight")
        self.add_input("w_systems", val=1.0, units="N",
                      desc="Systems weight")
        
        self.add_output("w_struct", units="N",
                       desc="Total structural weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        outputs["w_struct"] = (inputs["w_wing"] + inputs["w_fuselage"] + 
                             inputs["w_lg"] + inputs["w_nacelle"] +
                             inputs["w_engine"] + inputs["w_prop"] +
                             inputs["w_systems"])


class WeightGroup(om.Group):
    """Group containing all weight calculation components."""
    
    def setup(self):
        self.add_subsystem("wing_weight", WingWeight(), promotes_inputs=["*"])
        self.add_subsystem("fuselage_weight", FuselageWeight(), promotes_inputs=["*"])
        self.add_subsystem("landing_gear_weight", LandingGearWeight(), promotes_inputs=["*"])
        self.add_subsystem("engine_weight", EngineWeight(), promotes_inputs=["*"])
        self.add_subsystem("propeller_weight", PropellerWeight(), promotes_inputs=["*"])
        self.add_subsystem("nacelle_weight", NacelleWeight(), 
                          promotes_inputs=["n_en"],
                          promotes_outputs=["w_nacelle"])
        self.add_subsystem("systems_weight", SystemsWeight(), promotes_inputs=["*"])
        
        # Connect engine weight to nacelle component
        self.connect("engine_weight.w_engine", "nacelle_weight.w_engine")
        
        # Add aircraft weight calculator
        self.add_subsystem("aircraft_weight",
                          AircraftWeight(),
                          promotes_outputs=["w_struct"])
        
        # Connect all weights to aircraft weight calculator
        self.connect("wing_weight.w_wing", "aircraft_weight.w_wing")
        self.connect("fuselage_weight.w_fuselage", "aircraft_weight.w_fuselage")
        self.connect("landing_gear_weight.w_lg", "aircraft_weight.w_lg")
        self.connect("nacelle_weight.w_nacelle", "aircraft_weight.w_nacelle")
        self.connect("engine_weight.w_engine", "aircraft_weight.w_engine")
        self.connect("propeller_weight.w_prop", "aircraft_weight.w_prop")
        self.connect("systems_weight.w_systems", "aircraft_weight.w_systems")


if __name__ == "__main__":
    import openmdao.api as om
    
    # Create problem
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output("w_dg", val=50000.0, units="N")
    ivc.add_output("n_ult", val=3.8)
    ivc.add_output("s_w", val=25.0, units="m**2")
    ivc.add_output("ar_w", val=10.0)
    ivc.add_output("t_c_w", val=0.12)
    ivc.add_output("sweep_w", val=0.0, units="rad")
    ivc.add_output("l_f", val=12.0, units="m")
    ivc.add_output("d_f", val=1.5, units="m")
    ivc.add_output("v_p", val=100.0, units="m/s")
    ivc.add_output("n_l", val=3.0)
    ivc.add_output("l_m", val=0.9, units="m")
    ivc.add_output("n_en", val=2.0)
    ivc.add_output("p_to", val=1000000.0, units="W")
    ivc.add_output("n_p", val=2.0)
    ivc.add_output("d_p", val=2.0, units="m")
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('weights', WeightGroup(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Generate N2 diagram
    om.n2(prob)
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nStructural Weight Results:')
    print('Wing Weight:', prob.get_val('weights.wing_weight.w_wing')[0], 'N')
    print('Fuselage Weight:', prob.get_val('weights.fuselage_weight.w_fuselage')[0], 'N')
    print('Landing Gear Weight:', prob.get_val('weights.landing_gear_weight.w_lg')[0], 'N')
    print('Engine Weight:', prob.get_val('weights.engine_weight.w_engine')[0], 'N')
    print('Propeller Weight:', prob.get_val('weights.propeller_weight.w_prop')[0], 'N')
    print('Nacelle Weight:', prob.get_val('weights.nacelle_weight.w_nacelle')[0], 'N')
    print('Systems Weight:', prob.get_val('weights.systems_weight.w_systems')[0], 'N')
    print('Total Structural Weight:', prob.get_val('weights.w_struct')[0], 'N')