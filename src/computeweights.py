from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any


class WingWeight(om.ExplicitComponent):
    """Computes wing weight based on Raymer's equations."""
    
    def setup(self):
        self.add_input("w_mtow", val=1.0, units="lbm",
                      desc="Maximum takeoff weight")
        self.add_input("n_ult", val=1.0, units=None,
                      desc="Ultimate load factor")
        self.add_input("s_w", val=1.0, units="ft**2",
                      desc="Wing area")
        self.add_input("ar_w", val=1.0, units=None,
                      desc="Wing aspect ratio")
        self.add_input("t_c_w", val=1.0, units=None,
                      desc="Wing thickness-to-chord ratio")
        self.add_input("sweep_c_4_w", val=1.0, units="rad",
                      desc="Wing quarter-chord sweep angle")
        self.add_input("q_cruise", val=1.0, units="lbf/ft**2",
                      desc="Cruise dynamic pressure")
        self.add_input("lambda_w", val=1.0, units=None,
                      desc="Wing taper ratio")
        
        self.add_output("w_wing", units="lbm",
                       desc="Wing weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):

        w_mtow = inputs["w_mtow"] 
        n_ult = inputs["n_ult"]
        s_w = inputs["s_w"]
        ar_w = inputs["ar_w"]
        t_c_w = inputs["t_c_w"]
        sweep_c_4_w = inputs["sweep_c_4_w"]
        q_cruise = inputs["q_cruise"]
        lambda_w = inputs["lambda_w"]
        

        outputs["w_wing"] = (
        0.036 
        * s_w**0.758 
        * w_mtow**0.0035 
        * (ar_w / (np.cos(sweep_c_4_w)**2))**0.6 
        * q_cruise**0.006 
        * lambda_w**0.04 
        * ((100 * t_c_w) / np.cos(sweep_c_4_w))**-0.3 
        * (n_ult * w_mtow)**0.49
    )
        

class HorizontalTailWeight(om.ExplicitComponent):
    """Computes horizontal tail weight using Raymer's equation."""
    
    def setup(self):
        # Add inputs
        self.add_input('n_ult', val=3.8, units=None,
                      desc='Ultimate load factor')
        self.add_input('w_mtow', val=1000.0, units='lbm',
                      desc='Takeoff gross weight')
        self.add_input('q_cruise', val=100.0, units='lbf/ft**2',
                      desc='Dynamic pressure at cruise')
        self.add_input('s_ht', val=10.0, units='ft**2',
                      desc='Horizontal tail area')
        self.add_input('t_c_ht', val=0.12, units=None,
                      desc='Thickness-to-chord ratio')
        self.add_input('sweep_c_4_ht', val=0.0, units='rad',
                      desc='Horizontal tail quarter-chord sweep angle')
        self.add_input('ar_w', val=8.0, units=None,
                      desc='Wing aspect ratio')
        self.add_input('lambda_ht', val=0.4, units=None,
                      desc='Horizontal tail taper ratio')
        
        # Add output
        self.add_output('w_htail', units='lbm',
                       desc='Horizontal tail weight')
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):
        n_ult = inputs['n_ult']
        w_mtow = inputs['w_mtow']
        q_cruise = inputs['q_cruise']
        s_ht = inputs['s_ht']
        t_c_ht = inputs['t_c_ht']
        sweep_c_4_ht = inputs['sweep_c_4_ht']
        ar_w = inputs['ar_w']
        lambda_ht = inputs['lambda_ht']
        
        outputs['w_htail'] = (
            0.016 
            * (n_ult * w_mtow)**0.414
            * q_cruise**0.168
            * s_ht**0.896 
            * ((100 * t_c_ht) / np.cos(sweep_c_4_ht))**-0.12
            * (ar_w / (np.cos(sweep_c_4_ht)**2))**0.043
            * lambda_ht**-0.02
        )


class VerticalTailWeight(om.ExplicitComponent):
    """
    Computes vertical tail weight using Raymer's equation.
    """
    
    def setup(self):
        # Add inputs
        self.add_input('f_tail', val=0.0, units=None,
                      desc='Fraction of tail area relative to vertical tail')
        self.add_input('n_ult', val=3.8, units=None,
                      desc='Ultimate load factor')
        self.add_input('w_mtow', val=1000.0, units='lbm',
                      desc='Takeoff gross weight')
        self.add_input('q_cruise', val=100.0, units='lbf/ft**2',
                      desc='Dynamic pressure at cruise')
        self.add_input('s_vt', val=10.0, units='ft**2',
                      desc='Vertical tail area')
        self.add_input('t_c_vt', val=0.12, units=None,
                      desc='Thickness-to-chord ratio')
        self.add_input('sweep_c_4_vt', val=0.0, units='rad',
                      desc='Vertical tail quarter-chord sweep angle')
        self.add_input('ar_w', val=8.0, units=None,
                      desc='Wing aspect ratio')
        self.add_input('lambda_vt', val=0.4, units=None,
                      desc='Vertical tail taper ratio')
        
        # Add output
        self.add_output('w_vtail', units='lbm',
                       desc='Vertical tail weight')
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):

        f_tail = inputs['f_tail']
        n_ult = inputs['n_ult']
        w_mtow = inputs['w_mtow']
        q_cruise = inputs['q_cruise']
        s_vt = inputs['s_vt']
        t_c_vt = inputs['t_c_vt']
        sweep_c_4_vt = inputs['sweep_c_4_vt']
        ar_w = inputs['ar_w']
        lambda_vt = inputs['lambda_vt']
        
        outputs['w_vtail'] = (
            0.073
            * (1 + 0.2 * f_tail)
            * (n_ult * w_mtow)**0.376
            * q_cruise**0.122
            * s_vt**0.873
            * ((100 * t_c_vt) / np.cos(sweep_c_4_vt))**-0.49
            * (ar_w / (np.cos(sweep_c_4_vt)**2))**0.357
            * lambda_vt**0.039
        )
        

class ElectricalSystemWeight(om.ExplicitComponent):
    """Computes electrical system weight using Raymer/USAF equation."""
    
    def setup(self):
        self.add_input("w_fuel_system", val=100.0, units="lbm",
                      desc="Fuel system weight")
        self.add_input("w_avionics", val=100.0, units="lbm",
                      desc="Avionics system weight")
        
        self.add_output("w_electrical", units="lbm",
                       desc="Electrical system weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        w_fuel_system = inputs["w_fuel_system"]
        w_avionics = inputs["w_avionics"]
        
        outputs["w_electrical"] = 12.57 * (w_fuel_system + w_avionics)**0.51
        

class FurnishingsWeight(om.ExplicitComponent):
    """Computes furnishings weight using Raymer's equation."""
    
    def setup(self):
        self.add_input("w_mtow", val=2000.0, units="lbm",
                      desc="Takeoff gross weight")
        
        self.add_output("w_furnishings", units="lbm",
                       desc="Furnishings weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        
        w_mtow = inputs["w_mtow"]
        outputs["w_furnishings"] = 0.0582 * w_mtow - 65
        

class InstalledTurbineWeight(om.ExplicitComponent):
    """Computes installed turbine weight using Raymer's equation."""
    
    def setup(self):
        self.add_input("w_turbine", val=500.0, units="lbm",
                      desc="Weight of single turbine")
        self.add_input("n_turbine", val=2.0, units=None,
                      desc="Number of turbines")
        
        self.add_output("w_turbines", units="lbm",
                       desc="Total installed turbine engines weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        w_turbine = inputs["w_turbine"]
        n_turbine = inputs["n_turbine"]
        
        outputs["w_turbines"] = 2.575 * w_turbine**0.922 * n_turbine
    
class AvionicsWeight(om.ExplicitComponent):
    """Computes avionics system weight using Raymer's equation."""
    
    def setup(self):
        self.add_input("w_uav", val=100.0, units="lbm",
                      desc="Uninstalled avionics weight")
        
        self.add_output("w_avionics", units="lbm",
                       desc="Installed avionics system weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):

        w_uav = inputs["w_uav"]
        outputs["w_avionics"] = 2.11 * w_uav**0.933
        

class FlightControlWeight(om.ExplicitComponent):
    """Computes flight control system weight using Raymer's equation."""
    
    def setup(self):
        self.add_input("l_fs", val=20.0, units="ft",
                      desc="Fuselage length")
        self.add_input("b_w", val=30.0, units="ft",
                      desc="Wingspan")
        self.add_input("n_ult", val=3.8, units=None,
                      desc="Ultimate load factor")
        self.add_input("w_mtow", val=2000.0, units="lbm",
                      desc="Takeoff gross weight")
        
        self.add_output("w_fc_sys", units="lbm",
                       desc="Flight control system weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):


        l_fs = inputs["l_fs"]
        b_w = inputs["b_w"]
        n_ult = inputs["n_ult"]
        w_mtow = inputs["w_mtow"]
        
        outputs["w_fc_sys"] = (
            0.053 
            * l_fs**1.536 
            * b_w**0.371 
            * ((n_ult * w_mtow * 1e-4)**0.80)
        )
        

                      
class TurbineWeight(om.ExplicitComponent):
    """Computes gas turbine weight using empirical equation."""
    
    def setup(self):
        self.add_input("p_max", val=1000.0, units="hp",
                      desc="Maximum engine power")
        
        self.add_output("w_turbine", units="lbm",
                       desc="turbine engine weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        p_max = inputs["p_max"]
        outputs["w_turbine"] = 71.65 + 0.3658 * p_max
        

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
        self.add_input("w_motors", val=1.0, units="N",
                      desc="Motor weight")
        self.add_input("w_propellers", val=1.0, units="N",
                      desc="Propeller weight")
        
        self.add_output("w_nacelles", units="N",
                       desc="Nacelles weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):

        w_engine = inputs["w_motors"] + inputs["w_propellers"]
        
        outputs["w_nacelles"] = 0.6 * w_engine  * 9.81

class FuelSystemWeight(om.ExplicitComponent):
    """Computes fuel system weight using Raymer's equation."""
    
    def setup(self):
        self.add_input("Q_tot", val=100.0, units="galUS",
                      desc="Total fuel quantity")
        self.add_input("Q_int", val=80.0, units="galUS",
                      desc="Internal fuel quantity")
        self.add_input("N_TANK", val=2.0, units=None,
                      desc="Number of fuel tanks")
        self.add_input("N_ENG", val=2.0, units=None,
                      desc="Number of engines")
        
        self.add_output("w_fuel_system", units="lbm",
                       desc="Fuel system weight")
        
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):
        Q_tot = inputs["Q_tot"]
        Q_int = inputs["Q_int"]
        N_TANK = inputs["N_TANK"]
        N_ENG = inputs["N_ENG"]
        
        outputs["w_fuel_system"] = (
            2.49 
            * Q_tot**0.726 
            * (Q_tot / (Q_tot + Q_int))**0.363 
            * N_TANK**0.242 
            * N_ENG**0.157
        )
        

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
        self.add_input("w_nacelles", val=1.0, units="N",
                      desc="Nacelle weight")
        self.add_input("w_prop", val=1.0, units="N",
                      desc="Propeller weight")
        self.add_input("w_systems", val=1.0, units="N",
                      desc="Systems weight")
        self.add_input("w_ptrain", val=1.0, units="N",
                      desc="Powertrain weight")
        self.add_input("w_fuel", val=1.0, units="N",
                      desc="Fuel weight")
        self.add_input("w_payload", val=1.0, units="N",
                      desc="Payload weight")
        

        self.add_output("w_airframe", units="N",
                       desc="Total structural weight")
        
        self.add_output("w_mtow", units="N",
                       desc="Total maximum takeoff weight")
        

        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        outputs["w_airframe"] = (inputs["w_wing"] + inputs["w_fuselage"] + 
                             inputs["w_lg"] + inputs["w_nacelles"] +
                             inputs["w_prop"] +
                             inputs["w_systems"])
        outputs["w_mtow"] = (inputs["w_airframe"] + inputs["w_ptrain"] +
                             inputs["w_fuel"] + inputs["w_payload"])


class WeightGroup(om.Group):
    """Group containing all weight calculation components."""
    
    def setup(self):
        self.add_subsystem("wing_weight", WingWeight(), promotes_inputs=["*"])
        self.add_subsystem("fuselage_weight", FuselageWeight(), promotes_inputs=["*"])
        self.add_subsystem("landing_gear_weight", LandingGearWeight(), promotes_inputs=["*"])
        self.add_subsystem("propeller_weight", PropellerWeight(), promotes_inputs=["*"])
        self.add_subsystem("nacelle_weight", NacelleWeight(), 
                          promotes_inputs=["n_en"],
                          promotes_outputs=["w_nacelles"])
        self.add_subsystem("systems_weight", SystemsWeight(), promotes_inputs=["*"])
        
            
        # Add aircraft weight calculator
        self.add_subsystem("aircraft_weight",
                          AircraftWeight(),
                          promotes_outputs=["*"])
        
        # Connect all weights to aircraft weight calculator


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
    ivc.add_output("w_ptrain", val=20000.0, units="N")
    ivc.add_output("w_fuel", val=10000.0, units="N")
    ivc.add_output("w_payload", val=10000.0, units="N")
    
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
    print('Nacelle Weight:', prob.get_val('weights.nacelle_weight.w_nacelles')[0], 'N')
    print('Systems Weight:', prob.get_val('weights.systems_weight.w_systems')[0], 'N')
    print('Total Structural Weight:', prob.get_val('weights.w_airframe')[0], 'N')