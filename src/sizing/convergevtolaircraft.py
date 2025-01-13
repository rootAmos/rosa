from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any

from computehover import HoverEnergyGroup
from cruisemaxpower import CruiseMaxpower
from cruisemaxefficiency import CruiseMaxEfficiency

from computeweights import ComputeWeights
from computeamos import ComputeAtmos
from convergeturbocruiserange import ConvergeTurboCruiseRange


class ConvergeAircraft(om.Group):
    """
    Global sizing loop that converges aircraft MTOM through power, range, and weight calculations.
    """
    
    
    def setup(self):
        # Add hover power and energy calculation

        
        self.add_subsystem("atmos", ComputeAtmos(), 
                           promotes_inputs=["*"], 
                           promotes_outputs=["*"])

        self.add_subsystem("hover",
                          HoverEnergyGroup(),
                          promotes_inputs=["*"], promotes_outputs=["*"])
        

        
        self.add_subsystem("cruisemaxpow", CruiseMaxpower(), 
                           promotes_inputs=["cd0","e","s_ref","gamma","rho","w_mto","epsilon_r","ar_w","eta_fan","eta_duct","n_fans"], 
                           promotes_outputs=[])
        
        self.add_subsystem("cruisemaxeff", CruiseMaxEfficiency(), 
                           promotes_inputs=["cd0","e","s_ref","gamma","rho","w_mto","epsilon_r","ar_w","eta_fan","eta_duct","n_fans"], 
                           promotes_outputs=["eta_pt"])

        
        self.add_subsystem("cruiserange",
                          ConvergeTurboCruiseRange(),
                          promotes_inputs=["*"],
                          promotes_outputs=["*"])

        
        self.add_subsystem("weights",
                          ComputeWeights(),
                          promotes_outputs=["*"],
                          promotes_inputs=["*"])
        
        self.connect("cruisemaxeff.cruise.CL", "cl")
        self.connect("cruisemaxeff.cruise.CD", "cd")
        self.connect("w_mto_calc", "w_mto")
        #self.connect("n_props", "n_lift_props")

        # Max Efficiency Cruise
        #self.connect("cruisemaxeff.CL", "cruisemaxeff.w_mto")
        #self.connect("n_lift_props", "n_props")
        #self.connect("n_crz_fans", "n_fans")

        #self.connect("maxcrzpower.p_shaft_unit", "fan.p_fan_shaft_max")
        
        # Add nonlinear solver for MTOM convergence
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options['maxiter'] = 50
        self.nonlinear_solver.options['iprint'] = 2
        
        # Add linear solver
        self.linear_solver = om.DirectSolver()


if __name__ == "__main__":
    # Create problem
    prob = om.Problem(reports=False)

    ivc = om.IndepVarComp()

    
    q_cruise = 1.225 * 0.5 * 60**2

    # WingWeight inputs
    #ivc.add_output("w_mto", val=1.0, units="lbf", desc="Maximum takeoff weight")
    ivc.add_output("n_ult", val=3.75, units=None, desc="Ultimate load factor")
    ivc.add_output("s_ref", val=282.74, units="ft**2", desc="Wing area")
    ivc.add_output("ar_w", val=14.0, units=None, desc="Wing aspect ratio")
    ivc.add_output("t_c_w", val=0.13, units=None, desc="Wing thickness-to-chord ratio")
    ivc.add_output("sweep_c_4_w", val=10/180 * np.pi, units="rad", desc="Wing quarter-chord sweep angle")
    ivc.add_output("q_cruise", val=q_cruise, units="Pa", desc="Cruise dynamic pressure")
    ivc.add_output("lambda_w", val=0.45, units=None, desc="Wing taper ratio")
    ivc.add_output("b_w", val=43, units="ft", desc="Wingspan")

    # HorizontalTailWeight inputs
    # (uses n_ult, w_mto, q_cruise, ar_w from above)
    ivc.add_output("s_ht", val=30, units="ft**2", desc="Horizontal tail area")
    ivc.add_output("t_c_ht", val=0.17, units=None, desc="Horizontal tail thickness-to-chord ratio")
    ivc.add_output("sweep_c_4_ht", val=45/180 * np.pi, units="rad", desc="Horizontal tail quarter-chord sweep angle")
    ivc.add_output("lambda_ht", val=0.45, units=None, desc="Horizontal tail taper ratio")

    # VerticalTailWeight inputs
    # (uses n_ult, w_mto, q_cruise, ar_w from above)
    ivc.add_output("f_tail", val=1.0, units=None, desc="Tail shape factor")
    ivc.add_output("s_vt", val=36, units="ft**2", desc="Vertical tail area")
    ivc.add_output("t_c_vt", val=0.17, units=None, desc="Vertical tail thickness-to-chord ratio")
    ivc.add_output("sweep_c_4_vt", val=35/180 * np.pi, units="rad", desc="Vertical tail quarter-chord sweep angle")
    ivc.add_output("lambda_vt", val=0.45, units=None, desc="Vertical tail taper ratio")

    # AvionicsWeight inputs
    ivc.add_output("w_uav", val=100, units="lbf", desc="Uninstalled avionics weight")

    # FlightControlSystemWeight inputs
    # (uses n_ult, w_mto from above)
    ivc.add_output("l_fs", val=40, units="ft", desc="Fuselage structure length")
    ivc.add_output("d_fs", val=5, units="ft", desc="Fuselage structural diameter")
    ivc.add_output("l_ht", val=20, units="ft", desc="Horizontal tail lever arm length")
    ivc.add_output("v_fp", val=1e-6, units="ft**3", desc="Fuselage pressurized volume")
    ivc.add_output("delta_p", val=9, units="psi", desc="Fuselage pressurization delta")
    ivc.add_output("s_fus", val=650, units="ft**2", desc="Fuselage wetted area")

    # LandingGearWeight inputs
    # (uses w_mto, w_fuel from above)
    ivc.add_output("n_l", val=3.5, units=None, desc="Ultimate landing load factor")
    ivc.add_output("l_m", val=1, units="ft", desc="Length of main landing gear strut")
    ivc.add_output("l_n", val=1, units="ft", desc="Length of nose landing gear strut")

    # Electrical system inputs
    ivc.add_output("n_gens", val=2.0)

    ivc.add_output("w_pay", val=6*100.0, units="lbf", desc="Payload weight")
    
    # Power/energy densities
    ivc.add_output("sp_motor", val=5000.0, units="W/kg")
    ivc.add_output("sp_pe", val=10000.0, units="W/kg")
    ivc.add_output("sp_cbl", val=20000.0, units="W/kg")
    ivc.add_output("sp_bat", val=3000.0, units="W/kg")
    ivc.add_output("se_bat", val=200.0, units="W*h/kg")
    ivc.add_output("turbine_power_density", val=2000.0, units="W/kg")
    ivc.add_output("sp_gen", val=3000.0, units="W/kg")
    
    # Fuel system inputs
    ivc.add_output("v_fuel_tot", val=500.0, units="galUS")
    ivc.add_output("v_fuel_int", val=400.0, units="galUS")
    ivc.add_output("n_tank", val=2.0)
    
    # Turbine inputs
    
    # Propeller inputs
    n_lift_motors = 8
    n_crz_motors = 2
    n_motors = n_lift_motors + n_crz_motors

    ivc.add_output("n_props", val=n_lift_motors)
    ivc.add_output("d_prop_blades", val=2.5, units="m")
    ivc.add_output("d_prop_hub", val=0.5, units="m")
    ivc.add_output("k_prop", val=31.92)
    ivc.add_output("n_prop_blades", val=5.0)

    # Ducted fan inputs
    ivc.add_output("d_fan_blades", val=2.5, units="m")
    ivc.add_output("d_fan_hub", val=0.5, units="m")
    ivc.add_output("n_fans", val=n_crz_motors)
    ivc.add_output("k_fan", val=31.92)
    ivc.add_output("n_fan_blades", val=5.0)
    ivc.add_output("epsilon_r", val=1.2, units=None)

    # Performance Inputs
    ivc.add_output("eta_i", val=0.95, units=None)
    ivc.add_output("eta_m", val=0.95, units=None)
    ivc.add_output("eta_gen", val=0.95, units=None)
    ivc.add_output("eta_motor", val=0.95, units=None)
    ivc.add_output("eta_fan", val=0.95, units=None)
    ivc.add_output("eta_duct", val=0.95, units=None)
    ivc.add_output("eta_pe", val=0.95, units=None)
    ivc.add_output("eta_cable", val=0.95, units=None)


    ivc.add_output("cp", val=0.5/60/60/1000, units="kg/W/s")
    ivc.add_output("target_range", val=1030*1000.0, units="m")

    # Cruise
    ivc.add_output('cd0', val=0.02)
    ivc.add_output('e', val=0.8)
    ivc.add_output('gamma', val=0.0, units='rad')
    ivc.add_output('v_max_speed', val=100.0, units='m/s')
    ivc.add_output('v_max_range', val=60.0, units='m/s')
    ivc.add_output('alt', val=1000.0, units='ft')

    # Hover


    ivc.add_output('t_hover', val=120.0, units='s')
    ivc.add_output('epsilon_hvr', val=1.0, units=None)
    ivc.add_output('n_lift_motors', val=n_lift_motors)
    ivc.add_output('n_crz_motors', val=n_crz_motors)
    ivc.add_output("n_motors", val=n_motors)


    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('converge_aircraft', ConvergeAircraft(), promotes_inputs=["*"])


    model.connect("d_prop_blades", "propeller.d_blades")
    model.connect("k_prop", "propeller.k_prop")
    model.connect("n_prop_blades", "propeller.n_blades")
    model.connect("n_props", "propeller.n_props")
    model.connect("n_fans", "fan.n_props")

    model.connect("k_fan", "fan.k_prop")
    model.connect("n_fan_blades", "fan.n_blades")

    model.connect("d_fan_blades", ["fan.d_blades","cruisemaxpow.d_blades","cruisemaxeff.d_blades"])

    model.connect("d_fan_hub", ["cruisemaxpow.d_hub","cruisemaxeff.d_hub"])


    # Add connections for CruiseMaxpower
    model.connect("v_max_speed", "cruisemaxpow.vel")
    model.connect("v_max_range",  "cruisemaxeff.vel")

 



    # Set up 
    #model.connect("p_fan_shaft_max", "fan.p_shaft")
    #model.connect("k_fan", "fan.k_prop")
    #model.connect("n_fan_blades", "fan.n_blades")

    # Setup problem
    prob.setup()
    om.n2(prob, show_browser=True)
    
    # Run model
    prob.run_model()
    
    # Print results
    print("\nAircraft Sizing Results:")
    print("Final MTOM:", prob.get_val("converge.w_struct")[0]/9.81, "kg")
    print("Hover Power Required:", prob.get_val("converge.hover.electrical_power")[0]/1000, "kW")
    print("bat Energy:", prob.get_val("converge.hover.bat_energy")[0]/3.6e6, "kWh")
    print("Fuel Weight:", prob.get_val("converge.cruise.fuel_weight.w_fuel")[0]/9.81, "kg")
    print("Empty Weight Fraction:", prob.get_val("converge.weights.empty_weight_fraction")[0])
    
    # Check model
    prob.check_partials(compact_print=True) 