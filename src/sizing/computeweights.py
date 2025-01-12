from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any


class WingWeight(om.ExplicitComponent):
    """Computes wing weight based on Raymer's equations."""
    
    def setup(self):
        self.add_input("w_mto", val=1.0, units="lbf",
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
        
        self.add_output("w_wing", units="lbf",
                       desc="Wing weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):

        w_mto = inputs["w_mto"] 
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
        * w_mto**0.0035 
        * (ar_w / (np.cos(sweep_c_4_w)**2))**0.6 
        * q_cruise**0.006 
        * lambda_w**0.04 
        * ((100 * t_c_w) / np.cos(sweep_c_4_w))**-0.3 
        * (n_ult * w_mto)**0.49
    )
        

class HorizontalTailWeight(om.ExplicitComponent):
    """Computes horizontal tail weight using Raymer's equation."""
    
    def setup(self):
        # Add inputs
        self.add_input('n_ult', val=3.8, units=None,
                      desc='Ultimate load factor')
        self.add_input('w_mto', val=1000.0, units='lbf',
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
        self.add_output('w_htail', units='lbf',
                       desc='Horizontal tail weight')
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):
        n_ult = inputs['n_ult']
        w_mto = inputs['w_mto']
        q_cruise = inputs['q_cruise']
        s_ht = inputs['s_ht']
        t_c_ht = inputs['t_c_ht']
        sweep_c_4_ht = inputs['sweep_c_4_ht']
        ar_w = inputs['ar_w']
        lambda_ht = inputs['lambda_ht']
        
        outputs['w_htail'] = (
            0.016 
            * (n_ult * w_mto)**0.414
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
        self.add_input('w_mto', val=1000.0, units='lbf',
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
        self.add_output('w_vtail', units='lbf',
                       desc='Vertical tail weight')
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):

        f_tail = inputs['f_tail']
        n_ult = inputs['n_ult']
        w_mto = inputs['w_mto']
        q_cruise = inputs['q_cruise']
        s_vt = inputs['s_vt']
        t_c_vt = inputs['t_c_vt']
        sweep_c_4_vt = inputs['sweep_c_4_vt']
        ar_w = inputs['ar_w']
        lambda_vt = inputs['lambda_vt']
        
        outputs['w_vtail'] = (
            0.073
            * (1 + 0.2 * f_tail)
            * (n_ult * w_mto)**0.376
            * q_cruise**0.122
            * s_vt**0.873
            * ((100 * t_c_vt) / np.cos(sweep_c_4_vt))**-0.49
            * (ar_w / (np.cos(sweep_c_4_vt)**2))**0.357
            * lambda_vt**0.039
        )
        

class ElectricalSystemWeight(om.ExplicitComponent):
    """Computes electrical system weight using Raymer/USAF equation."""
    
    def setup(self):
        self.add_input("w_fuel_system", val=100.0, units="lbf",
                      desc="Fuel system weight")
        self.add_input("w_avionics", val=100.0, units="lbf",
                      desc="Avionics system weight")
        
        self.add_output("w_electrical", units="lbf",
                       desc="Electrical system weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        w_fuel_system = inputs["w_fuel_system"]
        w_avionics = inputs["w_avionics"]
        
        outputs["w_electrical"] = 12.57 * (w_fuel_system + w_avionics)**0.51
        

class FurnishingsWeight(om.ExplicitComponent):
    """Computes furnishings weight using Raymer's equation."""
    
    def setup(self):
        self.add_input("w_mto", val=2000.0, units="lbf",
                      desc="Takeoff gross weight")
        
        self.add_output("w_furnishings", units="lbf",
                       desc="Furnishings weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        
        w_mto = inputs["w_mto"]
        outputs["w_furnishings"] = 0.0582 * w_mto - 65
        


    
class AvionicsWeight(om.ExplicitComponent):
    """Computes avionics system weight using Raymer's equation."""
    
    def setup(self):
        self.add_input("w_uav", val=100.0, units="lbf",
                      desc="Uninstalled avionics weight")
        
        self.add_output("w_avionics", units="lbf",
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
        self.add_input("w_mto", val=2000.0, units="lbf",
                      desc="Takeoff gross weight")
        
        self.add_output("w_fc_sys", units="lbf",
                       desc="Flight control system weight")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):


        l_fs = inputs["l_fs"]
        b_w = inputs["b_w"]
        n_ult = inputs["n_ult"]
        w_mto = inputs["w_mto"]
        
        outputs["w_fc_sys"] = (
            0.053 
            * l_fs**1.536 
            * b_w**0.371 
            * ((n_ult * w_mto * 1e-4)**0.80)
        )
        

                      

        

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
        self.add_input("w_systems", val=1.0, units="N",
                      desc="Systems weight")
        self.add_input("w_ptrain", val=1.0, units="N",
                      desc="Powertrain weight")
        self.add_input("w_payload", val=1.0, units="N",
                      desc="Payload weight")
        

        self.add_output("w_airframe", units="N",
                       desc="Total structural weight")
        
        self.add_output("w_mto", units="N",
                       desc="Total maximum takeoff weight")
        

        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):
        outputs["w_airframe"] = (inputs["w_wing"] + inputs["w_fuselage"] + 
                             inputs["w_lg"] + inputs["w_systems"])
        outputs["w_mto"] = (inputs["w_airframe"] + inputs["w_ptrain"] +
                             inputs["w_payload"])


class WeightGroup(om.Group):
    """Group containing all weight calculation components."""
    
    def setup(self):
        self.add_subsystem("wing_weight", WingWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("fuselage_weight", FuselageWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("landing_gear_weight", LandingGearWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("systems_weight", SystemsWeight(), promotes_inputs=["*"], promotes_outputs=["*"])
        
            
        # Add aircraft weight calculator
        self.add_subsystem("aircraft_weight",
                          AircraftWeight(),
                          promotes_inputs=["*"],
                          promotes_outputs=["*"])
        
        # Connect all weights to aircraft weight calculator
        self.nonlinear_solver = om.NewtonSolver()
        self.nonlinear_solver.options['maxiter'] = 20
        self.nonlinear_solver.options['atol'] = 1e-6
        self.nonlinear_solver.options['rtol'] = 1e-6
        self.nonlinear_solver.options['solve_subsystems'] = True

        # Add linear solver
        self.linear_solver = om.DirectSolver()
    


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
    ivc.add_output("w_ptrain", val=25000.0, units="N")
    ivc.add_output("w_payload", val=10000.0, units="N")
    ivc.add_output("sweep_c_4_w", val=0.0, units="rad")
    ivc.add_output("lambda_w", val=0.5)
    ivc.add_output("q_cruise", val=100.0, units="lbf/ft**2")
    ivc.add_output("w_fuel_system", val=10000.0, units="N")
    
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