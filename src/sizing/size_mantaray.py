import openmdao.api as om
import numpy as np


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.sizing.mission.top_of_climb_req import TopOfClimbReq
from src.sizing.mission.cruise_req import CruiseRequirements
from src.sizing.weights.manta_weight import MantaWeight
from src.sizing.weights.ray_weight import RayWeight
from src.sizing.weights.mass_weight_conv import WeightToMass, MassToWeight
from src.sizing.ptrain.maxpowers import ExtractMaxPower

class MassToVolume(om.ExplicitComponent):
    """
    Computes volume from mass using density.
    """
    
    def setup(self):
        # Inputs
        self.add_input('mass', val=1.0, units='kg', desc='Mass')
        self.add_input('density', val=1000.0, units='kg/m**3', desc='Density')
        
        # Outputs
        self.add_output('volume', val=1.0, units='m**3', desc='Volume')
        
        # Declare partials
        self.declare_partials('volume', ['mass', 'density'])
        
    def compute(self, inputs, outputs):
        mass = inputs['mass']
        density = inputs['density']
        
        outputs['volume'] = mass / density
        
    def compute_partials(self, inputs, partials):
        density = inputs['density']
        
        partials['volume', 'mass'] = 1.0 / density
        partials['volume', 'density'] = -inputs['mass'] / (density**2)


class BatteryMassFractionMargin(om.ExplicitComponent):
    """
    Computes battery mass fraction margin: max_fraction - (m_batt / m_aircraft)
    Positive margin means we're under the maximum allowed fraction
    Negative margin means we've exceeded the maximum allowed fraction
    """
    
    def setup(self):
        # Inputs
        self.add_input('m_batt', val=1.0, units='kg',
                      desc='Battery mass')
        self.add_input('m_aircraft', val=1.0, units='kg',
                      desc='Total aircraft mass')
        self.add_input('max_fraction', val=0.3,
                      desc='Maximum allowable battery mass fraction')
        
        # Outputs
        self.add_output('fraction_margin', val=0.1,
                      desc='Battery mass fraction margin')
        
        # Declare partials
        self.declare_partials('fraction_margin', ['m_batt', 'm_aircraft', 'max_fraction'])
        
    def compute(self, inputs, outputs):
        m_batt = inputs['m_batt']
        m_aircraft = inputs['m_aircraft']
        max_fraction = inputs['max_fraction']
        
        outputs['fraction_margin'] = max_fraction - (m_batt / m_aircraft)
        
    def compute_partials(self, inputs, partials):
        m_batt = inputs['m_batt']
        m_aircraft = inputs['m_aircraft']
        
        partials['fraction_margin', 'm_batt'] = -1.0 / m_aircraft
        partials['fraction_margin', 'm_aircraft'] = m_batt / m_aircraft**2
        partials['fraction_margin', 'max_fraction'] = 1.0

class MTOMMargin(om.ExplicitComponent):
    """Computes margin between specified MTOM and calculated MTOM"""
    
    def setup(self):
        self.add_input('w_mto', val=1.0, units='N', desc='Specified maximum takeoff weight')
        self.add_input('w_mto_calc', val=1.0, units='N', desc='Calculated maximum takeoff weight')
        
        self.add_output('mtom_margin', val=1.0, units='N', desc='Margin between specified and calculated MTOM')
        
    def setup_partials(self):
        self.declare_partials('mtom_margin', ['w_mto', 'w_mto_calc'])
        
    def compute(self, inputs, outputs):
        outputs['mtom_margin'] = abs(inputs['w_mto'] - inputs['w_mto_calc'])
        

    def compute_partials(self, inputs, partials):
        partials['mtom_margin', 'w_mto'] = 1.0
        partials['mtom_margin', 'w_mto_calc'] = -1.0


class MantaRayMassRatio(om.ExplicitComponent):
    def setup(self):
        self.add_input('ray_mass_ratio', val=1.0, desc='Ray MTO / Manta MTO ratio')
        self.add_input('manta_mto', val=0.0, units='kg', desc='Manta maximum takeoff mass')
        
        self.add_output('ray_mto', units='kg', desc='Ray maximum takeoff mass')
        self.add_output('mantaray_mto', units='kg', desc='Combined Manta and Ray maximum takeoff mass')

    def setup_partials(self):
        # Declare all partial derivatives
        self.declare_partials('*', '*')
    
    def compute(self, inputs, outputs):
        outputs['ray_mto'] = inputs['ray_mass_ratio'] * inputs['manta_mto']
        outputs['mantaray_mto'] = outputs['ray_mto'] + inputs['manta_mto']

    def compute_partials(self, inputs, partials):
        
        partials['ray_mto', 'ray_mass_ratio'] = inputs['manta_mto']
        partials['ray_mto', 'manta_mto'] = inputs['ray_mass_ratio']

        partials['mantaray_mto', 'ray_mass_ratio'] = inputs['manta_mto']
        partials['mantaray_mto', 'manta_mto'] = inputs['ray_mass_ratio'] + 1.0


class MantaCruiseFuel(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('g', default=9.806, desc='Gravity constant in m/s^2')

    def setup(self):
        self.add_input('m_fuel_mission', val=0.0, units='kg', desc='Total fuel mass manta')
        self.add_input('m_fuel_clmb', val=0.0, units='kg', desc='Fuel mass consumed during climb')
        

        self.add_output('w_fuel_crz', units='N', desc='Remaining fuel weight for cruise')

    def setup_partials(self):
        self.declare_partials('w_fuel_crz', '*')
    

    def compute(self, inputs, outputs):
        g = self.options['g']
        w_fuel_crz = inputs['m_fuel_mission'] - inputs['m_fuel_clmb']
        outputs['w_fuel_crz'] = w_fuel_crz * g
        
    def compute_partials(self, inputs, partials):
        g = self.options['g']
        # d(cruise_fuel_weight)/d(total_fuel_mass) = g
        partials['w_fuel_crz', 'm_fuel_mission'] = g
        # d(cruise_fuel_weight)/d(climb_fuel_mass) = -g
        partials['w_fuel_crz', 'm_fuel_clmb'] = -g



class SizeMantaRay(om.Group):
    """
    Group that sizes the Manta-Ray aircraft system:
    1. Analyzes top of climb requirements (both vehicles)
    2. Analyzes cruise requirements (Manta only)
    3. Determines power system requirements
    4. Estimates component masses
    """
    
    def initialize(self):
        self.options.declare('N_climb', default=1, desc='Number of nodes in climb')
        self.options.declare('N_cruise', default=1, desc='Number of nodes in cruise')
    
    def setup(self):
        N_climb = self.options['N_climb']
        N_cruise = self.options['N_cruise']

        self.add_subsystem("manta_ray_mass_ratio",
                          MantaRayMassRatio(),
                          promotes_inputs=["*"],
                          promotes_outputs=["*"])
        
        self.connect('climb.manta_fuel_mass', 'manta_fuel.m_fuel_clmb')
        self.connect('manta_fuel.w_fuel_crz', 'cruise.w_fuel')
        self.connect('ray_mto','ray_battery_mass_fraction.m_aircraft')


        self.add_subsystem('manta_fuel_volume', MassToVolume(), promotes_inputs=[], promotes_outputs=[])

        self.connect('manta_fuel_volume.volume', 'manta_weight.v_fuel_tot')


        self.add_subsystem('manta_ray_m2w', MassToWeight(g=9.806), promotes_inputs=[], promotes_outputs=[])



        # Add climb phase analysis



        self.add_subsystem('climb',
                          TopOfClimbReq(N=N_climb),
                          promotes_inputs=[],
                          promotes_outputs=[])

        

        self.add_subsystem("manta_fuel",
                          MantaCruiseFuel(),
                          promotes_inputs=[],
                          promotes_outputs=[])
        

        self.add_subsystem('manta_m2w', MassToWeight(g=9.806), promotes_inputs=[], promotes_outputs=[])
        self.add_subsystem('ray_m2w', MassToWeight(g=9.806), promotes_inputs=[], promotes_outputs=[])
        self.add_subsystem('manta_fuel_m2w', MassToWeight(g=9.806), promotes_inputs=[], promotes_outputs=[])



        self.connect('ray_mto', 'ray_m2w.mass')
        self.connect('ray_m2w.weight', 'ray_weight.w_mto')
        self.connect('manta_fuel_m2w.weight', 'manta_weight.w_fuel')



        self.connect('mantaray_mto', ['manta_ray_m2w.mass','climb.lift_req.mto'])
        self.connect('manta_ray_m2w.weight', ['climb.w_mto','mantaray_mtom_margin.w_mto'])
        self.connect('manta_m2w.weight', ['cruise.w_mto','manta_weight.w_mto'])







        #self.connect('manta_weight.w_fuel', 'manta_cruise_fuel.m_fuel')

        # Add cruise phase analysis
        self.add_subsystem('cruise',
                          CruiseRequirements(N=N_cruise),
                          promotes_inputs=[],
                          promotes_outputs=[])
        

        # Extract max powers
        self.connect('climb.propeller.p_shaft_unit','p_shaft_ray_climb')
        self.connect('climb.ductedfan.p_shaft_unit','p_shaft_manta_climb')
        self.connect('cruise.p_shaft_unit','p_shaft_manta_cruise')
        self.connect('climb.ray_elec_power','ray_elec_power_climb')
        self.connect('climb.manta_elec_power',['manta_elec_power_climb'])

        self.connect('p_motor_manta_unit_max',['manta_weight.p_motor','manta_weight.p_fan_shaft_max','manta_weight.compute_fan_weight.p_shaft'])
        self.connect('p_gen_unit_max','manta_weight.p_gen_unit')
        self.connect('p_turbine_unit_max','manta_weight.p_turbine_unit')
        self.connect('p_total_elec_power_climb_max',['ray_weight.p_elec'])
        self.connect('p_motor_ray_unit_max',['ray_weight.p_motor','ray_weight.prop_weight.p_shaft'])

        self.connect('p_batt_max',['ray_weight.p_bat'])
        self.connect('climb.ray_battery_energy','ray_weight.e_bat')
        self.connect('p_manta_elec_power_cruise_max','manta_weight.p_elec')

        
        self.add_subsystem("max_powers",
                          ExtractMaxPower(N_climb=N_climb, N_cruise=N_cruise),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])




        self.add_subsystem("manta_weight",
                          MantaWeight(),
                          promotes_outputs=[],
                          promotes_inputs=[])



        self.add_subsystem("ray_weight",
                          RayWeight(),
                          promotes_outputs=[],
                          promotes_inputs=[])
        
        self.add_subsystem('ray_mbat_w2m', WeightToMass(g=9.806), promotes_inputs=[], promotes_outputs=[])
        self.add_subsystem('manta_mbat_w2m', WeightToMass(g=9.806), promotes_inputs=[], promotes_outputs=[])


        self.connect('manta_weight.hybrid_ptrain_weight.compute_elec_ptrain_weight.w_bat', 'manta_mbat_w2m.weight')
        self.connect('ray_weight.w_bat', 'ray_mbat_w2m.weight')

        self.connect('manta_mbat_w2m.mass', 'manta_battery_mass_fraction.m_batt')
        self.connect('ray_mbat_w2m.mass', 'ray_battery_mass_fraction.m_batt')

        
        self.add_subsystem("manta_battery_mass_fraction",
                          BatteryMassFractionMargin(),
                          promotes_outputs=[],
                          promotes_inputs=[])
        
        self.add_subsystem("ray_battery_mass_fraction",
                          BatteryMassFractionMargin(),
                          promotes_outputs=[],
                          promotes_inputs=[])
        
        adder = om.AddSubtractComp()
        adder.add_equation('manta_ray_mto_calc',
                        [
                        'manta_mto_calc',
                        'ray_mto_calc',
                    ],
                        desc='Manta Ray MTOM',

                        vec_size=1, units="N")
        self.add_subsystem('sum_MTOW', adder, promotes=[])


        # Add MTOM margin check
        self.add_subsystem("mantaray_mtom_margin",
                          MTOMMargin(),
                          promotes_inputs=[],
                          promotes_outputs=[])
        
        self.connect('manta_weight.w_mto_calc', 'sum_MTOW.manta_mto_calc')
        self.connect('ray_weight.mto_calc', 'sum_MTOW.ray_mto_calc')


        #self.set_input_defaults("w_fuel", val=1.0, units="N")

        self.connect('sum_MTOW.manta_ray_mto_calc', 'mantaray_mtom_margin.w_mto_calc')


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Independent variable component
    ivc = om.IndepVarComp()
    
    N_climb = 10
    N_cruise = 10
    
    # Mission parameters
    ivc.add_output('u_eas', val=120, units='m/s', desc='Equivalent airspeed')
    ivc.add_output('u_cruise', val=200 * np.ones(N_cruise), units='m/s', desc='Cruise airspeed')
    ivc.add_output('gamma_climb', val=5.0 * np.ones(N_climb), units='deg', desc='Climb angle')

    ivc.add_output('gamma_cruise', val=0.0 * np.ones(N_cruise), units='deg', desc='Cruise angle')

    ivc.add_output('mto_manta', val=30000.0, units='kg', desc='Maximum takeoff weight')
    ivc.add_output('duration', val=600.0, units='s', desc='Climb duration')

    ivc.add_output('rho_fuel', val=800.0, units='kg/m**3', desc='Fuel density')

    ivc.add_output('manta_bat_mass_frac', val=0.15)
    ivc.add_output('ray_bat_mass_frac', val=0.55)
    ivc.add_output('z0', val=1500, units='ft', desc='Initial altitude')
    z1 = 30000
    ivc.add_output('z1', val=z1, units='ft', desc='Final altitude')
    ivc.add_output('z_cruise', val=z1 * np.ones(N_cruise), units='ft', desc='Cruise altitude')

    # Backup Battery for manta
    ivc.add_output('manta_p_bat', val=100e3,units='W' ,desc='Manta battery power')
    ivc.add_output('manta_e_bat', val=20*3.6e6,units='J' ,desc='Manta battery energy')

    #ivc.add_output('mu', val=1.789e-5 * np.ones(N_climb), units='Pa*s', desc='Dynamic viscosity')   



    ivc.add_output('mu_cruise', val=1.789e-5 * np.ones(N_cruise), units='Pa*s', desc='Dynamic viscosity')
    ivc.add_output('mu_climb', val=1.789e-5 * np.ones(N_climb), units='Pa*s', desc='Dynamic viscosity')
    ivc.add_output('ray_energy_ratio', val=0.99) # fraction of energy provided by ray in climb

    ivc.add_output('mto', val=10000.0, units='kg', desc='Maximum takeoff mass')
    ivc.add_output('cp', val=0.3/60/60/1000, units='kg/W/s')
    ivc.add_output('m_fuel_manta_mission', val=4000.0, units='kg')
    ivc.add_output('ray_mass_ratio', val=0.4, desc='Ray MTOM / Manta MTOM ratio')

    # Mission parameters
    ivc.add_output('target_range', val=1000.0, units='km')


    # Thrust power group inputs
    ivc.add_output('thrust_ratio', val=0.4 * np.ones(N_climb), desc='Ratio between ray and manta thrust')
    ivc.add_output('num_pods', val=6, desc='Number of ray propulsors')
    ivc.add_output('num_ducts', val=4, desc='Number of manta propulsors')

    # Ducted fan inputs
    ivc.add_output('d_blades_duct', 2.4, units='m')
    ivc.add_output('d_hub_duct', 0.5, units='m')
    ivc.add_output('epsilon_r', 1.5, units=None)
    ivc.add_output('eta_fan', 0.9 , units=None)
    ivc.add_output('eta_duct', 0.9, units=None)
    ivc.add_output('c_duct', val=1.0, units='m', desc='Nacelle length')
    ivc.add_output('od_duct', val=2.402, units='m', desc='Duct outer diameter')
    ivc.add_output('id_duct', val=2.401, units='m', desc='Duct inner diameter')
    ivc.add_output('Q_duct', val=1.0, desc='Duct interference factor')
    ivc.add_output('k_lam_duct', val=0.1, desc='Duct laminar flow fraction')
    ivc.add_output("n_blades_fan", val=12.0)
    ivc.add_output("k_fan", val=31.92)


    
    

    # Propeller inputs
    ivc.add_output('d_blades_prop', 3, units='m')
    ivc.add_output("n_blades_prop", val=5.0)

    ivc.add_output('d_hub_prop', 0.5, units='m')
    ivc.add_output('eta_prop', 0.9 , units=None)
    ivc.add_output('l_pod', val=2.0, units='m', desc='Pod length')
    ivc.add_output('d_pod', val=0.5, units='m', desc='Pod diameter')
    ivc.add_output('Q_pod', val=1.0, desc='Pod interference factor')
    ivc.add_output('k_lam_pod', val=0.1, desc='Pod laminar flow fraction')

    #ivc.add_output("n_prplsrs", val=2.0)
    ivc.add_output("k_prplsr", val=31.92)

    # Ducted Fan inputs
    #ivc.add_output("p_fan_shaft_max", val=250000.0, units="W")



    # Universal Powertrain Paramaeters
    ivc.add_output('eta_motor_ray', val=0.95)
    ivc.add_output('eta_motor_manta', val=0.95)
    ivc.add_output('eta_inverter_ray', val=0.95)
    ivc.add_output('eta_inverter_manta', val=0.95)
    ivc.add_output('eta_cable_ray', val=0.98)
    ivc.add_output('eta_cable_manta', val=0.98)
    ivc.add_output('eta_battery', val=0.95)
    ivc.add_output('eta_generator', val=0.90)
    ivc.add_output('eta_gen_inverter', val=0.95)
    ivc.add_output('psfc', val=2.0e-7, units='kg/J')

    # Manta Aero Parameters
    ivc.add_output('phi_50_manta', val=18.0, units='deg', desc='Manta 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil_manta', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')
    ivc.add_output('aspect_ratio_manta', val=10.0, desc='Manta aspect ratio')
    ivc.add_output('d_eps_manta_d_alpha', val=0.925, desc='Manta downwash derivative')
    ivc.add_output('alpha0_airfoil_manta', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_twist_manta', val=0.0, units='rad', desc='twist angle')
    ivc.add_output('sweep_25_manta', val=18.0, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('Q_manta', val=1.1, desc='Interference factor')
    ivc.add_output('S_exp_manta', val=120.0, units='m**2', desc='Exposed planform area')
    ivc.add_output('t_c_manta', val=0.19, desc='Thickness to chord ratio')
    ivc.add_output('tau_manta', val=0.8, desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
    ivc.add_output('lambda_manta', val=0.1, desc='Wing taper ratio')
    ivc.add_output('k_lam_manta', val=0.1, desc='Laminar flow fraction')
    ivc.add_output('sweep_max_t_manta', val=10,units='deg', desc='Wing sweep at maximum thickness')
    ivc.add_output('l_char_manta', val=7.6, units='m',desc='Characteristic length')
    ivc.add_output('h_winglet_manta', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('span_manta', val=30.0, units='m', desc='Wing span')
    ivc.add_output('k_WL_manta', val=2.83, desc='Winglet effectiveness factor')
    ivc.add_output('S_manta', val=120.0, units='m**2', desc='Manta reference area')


    # Ray Aero Parameters
    ivc.add_output('S_ray', val=89.0, units='m**2', desc='Ray area')
    ivc.add_output('phi_50_ray', val=5.0, units='deg', desc='Ray 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil_ray', val=2*np.pi, units='1/rad', desc='Ray airfoil lift curve slope')
    ivc.add_output('aspect_ratio_ray', val=10.0, desc='Ray aspect ratio')
    ivc.add_output('d_eps_ray_d_alpha', val=0.05, desc='Ray downwash derivative')
    ivc.add_output('alpha0_airfoil_ray', val=-2.0, units='deg', desc='Ray airfoil zero-lift angle')
    ivc.add_output('d_twist_ray', val=0.0, units='rad', desc='twist angle')
    ivc.add_output('sweep_25_ray', val=12.0, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('Q_ray', val=1.1, desc='Interference factor')
    ivc.add_output('S_exp_ray', val=50.0, units='m**2', desc='Exposed planform area')
    ivc.add_output('t_c_ray', val=0.19, desc='Thickness to chord ratio')
    ivc.add_output('tau_ray', val=0.8, desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
    ivc.add_output('lambda_ray', val=0.3, desc='Wing taper ratio')
    ivc.add_output('k_lam_ray', val=0.1, desc='Laminar flow fraction')
    ivc.add_output('sweep_max_t_ray', val=10,units='deg', desc='Wing sweep at maximum thickness')
    ivc.add_output('l_char_ray', val=1.28, units='m',desc='Characteristic length')
    ivc.add_output('h_winglet_ray', val=1e-3, units='m', desc='Height above ground')
    ivc.add_output('span_ray', val=26.0, units='m', desc='Wing span')
    ivc.add_output('k_WL_ray', val=2.83, desc='Winglet effectiveness factor')

    ## Mass Inputs
        # WingWeight inputs
    #ivc.add_output("w_mto", val=1.0, units="lbf", desc="Maximum takeoff weight")
    q_cruise = 1.225 * 0.5 * 80**2
    ivc.add_output("q_cruise", val=q_cruise, units="Pa", desc="Cruise dynamic pressure")

    

    # HorizontalTailWeight inputs
    # AvionicsWeight inputs
    ivc.add_output("w_uav_manta", val=500, units="lbf", desc="Uninstalled avionics weight")
    ivc.add_output("w_uav_ray", val=100, units="lbf", desc="Uninstalled avionics weight")

    ivc.add_output("n_ult", val=3.75, desc="Ultimate load factor")
    # FlightControlSystemWeight inputs
    # (uses n_ult, w_mto from above)
    ivc.add_output("l_fs", val=40, units="ft", desc="Fuselage structure length")

    # LandingGearWeight inputs
    # (uses w_mto, w_fuel from above)
    ivc.add_output("n_l", val=3.5, units=None, desc="Ultimate landing load factor")
    ivc.add_output("l_m", val=1, units="ft", desc="Length of main landing gear strut")
    ivc.add_output("l_n", val=1, units="ft", desc="Length of nose landing gear strut")

    # Electrical system inputs
    ivc.add_output("n_motors", val=8.0)
    ivc.add_output("n_gens", val=2.0)
    ivc.add_output("n_turbines", val=2.0)

    ivc.add_output("manta_lambda_aft", val=0.7)

    ivc.add_output("w_pay", val=80*215.0, units="lbf", desc="Payload weight")
    
    # Power/energy densities
    ivc.add_output("sp_motor", val=5000.0, units="W/kg")
    ivc.add_output("sp_pe", val=10000.0, units="W/kg")
    ivc.add_output("sp_cbl", val=20000.0, units="W/kg")
    ivc.add_output("sp_bat", val=3000.0, units="W/kg")
    ivc.add_output("se_bat", val=200.0, units="W*h/kg")
    ivc.add_output("turbine_power_density", val=2000.0, units="W/kg")
    ivc.add_output("sp_gen", val=3000.0, units="W/kg")
    
    # Fuel system inputs
    #v_fuel_gal = 100
    #ivc.add_output("v_fuel_tot", val=v_fuel_gal, units="galUS")
    ivc.add_output("v_fuel_int", val=1e-9, units="galUS")
    ivc.add_output("n_tank", val=2.0)
    
    ivc.add_output('eta_pe', val=0.95)
    ivc.add_output('eta_motor', val=0.95)
    ivc.add_output('eta_gen', val=0.95)
    # Propeller inputs

    # Fuselage dimensions
    ivc.add_output("l_cabin", val=8 *3.084, units="ft")
    ivc.add_output("w_cabin", val=19.6, units="ft")
    ivc.add_output("l_aft_fuselage", val=3*3.048, units="ft")
    ivc.add_output("w_aft_fuselage", val=19.6, units="ft")

    
    # Mission parameters
    
    # Build the model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('size', 
                            SizeMantaRay(N_climb=N_climb, N_cruise=N_cruise),
                            promotes=['*'])
    

    prob.model.connect('w_uav_manta', 'ray_weight.w_uav')
    prob.model.connect('w_uav_ray', 'manta_weight.w_uav')

    prob.model.connect('v_fuel_int', 'manta_weight.v_fuel_int')
    prob.model.connect('m_fuel_manta_mission', ['manta_fuel_volume.mass','manta_fuel.m_fuel_mission','manta_fuel_m2w.mass'])


    prob.model.connect('manta_p_bat', 'manta_weight.p_bat')
    prob.model.connect('manta_e_bat', 'manta_weight.e_bat')
    prob.model.connect('w_pay', 'manta_weight.w_pay')
    prob.model.connect('manta_lambda_aft', 'manta_weight.lambda_aft')

    prob.model.connect('rho_fuel', 'manta_fuel_volume.density')


    prob.model.connect('d_blades_duct', 'manta_weight.compute_fan_weight.d_blades')
    prob.model.connect('d_blades_prop', 'ray_weight.prop_weight.d_blades')
    prob.model.connect('n_blades_prop', 'ray_weight.prop_weight.n_blades')
    prob.model.connect('n_blades_fan', 'manta_weight.compute_fan_weight.n_blades')

    prob.model.connect("l_m", ["manta_weight.l_m", "ray_weight.l_m"])

    prob.model.connect("l_n", ["manta_weight.l_n", "ray_weight.l_n"])
    prob.model.connect("n_l", ["manta_weight.n_l", "ray_weight.n_l"])
    prob.model.connect("n_ult", ["manta_weight.n_ult", "ray_weight.n_ult"])

    prob.model.connect("num_ducts", ["manta_weight.n_motors","manta_weight.num_ducts","manta_weight.compute_fan_weight.n_prplsrs"])

    prob.model.connect("q_cruise", ["manta_weight.q_cruise","ray_weight.q_cruise"])

    prob.model.connect("l_cabin", "manta_weight.cabin_area.length")
    prob.model.connect("w_cabin", "manta_weight.cabin_area.width")
    prob.model.connect("l_aft_fuselage", "manta_weight.aft_fuse_area.length")
    prob.model.connect("w_aft_fuselage", "manta_weight.aft_fuse_area.width")

    


    prob.model.connect('sp_motor', ['manta_weight.sp_motor','ray_weight.sp_motor'])
    prob.model.connect('sp_pe', ['manta_weight.sp_pe','ray_weight.sp_pe'])
    prob.model.connect('sp_cbl', ['manta_weight.sp_cbl','ray_weight.sp_cbl'])
    prob.model.connect('sp_bat', ['manta_weight.sp_bat','ray_weight.sp_bat'])
    prob.model.connect('se_bat', ['manta_weight.se_bat','ray_weight.se_bat'])
    prob.model.connect('sp_gen', ['manta_weight.sp_gen'])
    prob.model.connect('turbine_power_density', ['manta_weight.turbine_power_density'])
    prob.model.connect('n_gens', ['manta_weight.n_gens'])
    prob.model.connect('n_tank', ['manta_weight.n_tank'])
    


    prob.model.connect('manta_bat_mass_frac', 'manta_battery_mass_fraction.max_fraction')
    prob.model.connect('ray_bat_mass_frac', 'ray_battery_mass_fraction.max_fraction')
    prob.model.connect('thrust_ratio', 'climb.thrust_ratio')
    prob.model.connect('u_cruise', 'cruise.u')
    prob.model.connect('z_cruise', 'cruise.alt')


    prob.model.connect('sweep_25_ray', ['climb.ray.sweep_25','ray_weight.sweep_c_4_w'])
    prob.model.connect('sweep_25_manta', ['cruise.sweep_25','climb.manta.sweep_25','manta_weight.sweep_c_4_w'])


    prob.model.connect('mto_manta', ['cruise.weight_mto.mass','manta_mto','cruise.lift_req.mto','manta_m2w.mass','manta_battery_mass_fraction.m_aircraft'])


    prob.model.connect('mu_climb', 'climb.mu')
    prob.model.connect('mu_cruise', 'cruise.mu')

    prob.model.connect('gamma_climb', 'climb.gamma')

    prob.model.connect('gamma_cruise', 'cruise.gamma')
    prob.model.connect('aspect_ratio_manta', ['climb.manta.aspect_ratio','manta_weight.ar_w'])
    prob.model.connect('aspect_ratio_ray', ['climb.ray.aspect_ratio','ray_weight.ar_w'])
    prob.model.connect('duration', 'climb.duration')



    prob.model.connect('d_eps_manta_d_alpha', 'climb.d_eps_manta_d_alpha')
    prob.model.connect('d_eps_ray_d_alpha', 'climb.d_eps_ray_d_alpha')



    prob.model.connect('phi_50_manta', ['climb.cl_alpha_manta.phi_50','cruise.phi_50'])
    prob.model.connect('phi_50_ray', 'climb.cl_alpha_ray.phi_50')


    prob.model.connect('cl_alpha_airfoil_manta', ['climb.cl_alpha_manta.cl_alpha_airfoil','cruise.cl_alpha_airfoil'])
    prob.model.connect('cl_alpha_airfoil_ray', 'climb.cl_alpha_ray.cl_alpha_airfoil')


    prob.model.connect('alpha0_airfoil_manta', ['climb.manta_cl0.alpha0_airfoil','cruise.alpha0_airfoil'])
    prob.model.connect('alpha0_airfoil_ray', 'climb.ray_cl0.alpha0_airfoil')


    prob.model.connect('d_twist_manta', ['climb.manta_cl0.d_twist','cruise.d_twist'])
    prob.model.connect('d_twist_ray', 'climb.ray_cl0.d_twist')

    prob.model.connect('cp','cruise.cp')

    prob.model.connect('aspect_ratio_manta', ['climb.cl_alpha_manta.aspect_ratio','cruise.aspect_ratio'])  
    prob.model.connect('aspect_ratio_ray', 'climb.cl_alpha_ray.aspect_ratio')

    prob.model.connect('l_char_manta', 'cruise.wing.l_char')

    prob.model.connect('Q_manta', ['climb.manta.wing.Q','cruise.wing.Q'])
    prob.model.connect('Q_duct', 'climb.manta.ducts.Q')
    prob.model.connect('Q_ray', 'climb.ray.canard.Q')
    prob.model.connect('Q_pod', 'climb.ray.pods.Q')


    prob.model.connect('S_manta', ['climb.S_manta','climb.S_ref','climb.manta.S_exp','cruise.S_ref',
                                   'cruise.S_exp','manta_weight.s_ref'])
    prob.model.connect('S_ray', ['climb.S_ray','climb.ray.S_exp','ray_weight.s_ref'])


    prob.model.connect('num_ducts', 'climb.num_ducts')
    prob.model.connect('num_pods', ['climb.num_pods','ray_weight.n_motors','ray_weight.prop_weight.n_prplsrs'])


    #prob.model.connect('climb_mu', 'climb.mu')
    prob.model.connect('eta_fan', 'climb.ductedfan.eta_fan')

    #prob.model.connect('d_hub_duct', 'climb.d_hub')
    #prob.model.connect('d_hub_prop', 'climb.d_hub_prop')

    prob.model.connect('z0', 'climb.z0')
    #prob.model.connect('w_mto', 'climb.w_mto')
    prob.model.connect('z1', 'climb.z1')
    prob.model.connect('u_eas', 'climb.u_eas')

    prob.model.connect('num_ducts', 'climb.ductedfan.num_ducts')
    prob.model.connect('num_pods', 'climb.propeller.n_motors')






    prob.model.connect('d_blades_duct',['climb.ductedfan.d_blades','cruise.d_blades'])
    prob.model.connect('d_hub_duct',['climb.ductedfan.d_hub','cruise.d_hub'])
    prob.model.connect('epsilon_r',['climb.ductedfan.epsilon_r','cruise.epsilon_r'])


    prob.model.connect('eta_duct','climb.ductedfan.eta_duct')


    prob.model.connect('d_blades_prop','climb.propeller.d_blades')
    prob.model.connect('d_hub_prop','climb.propeller.d_hub')
    prob.model.connect('eta_prop','climb.propeller.eta_prop')
    prob.model.connect('Q_duct','cruise.ducts.Q')





    prob.model.connect('l_char_manta', 'climb.manta.wing.l_char')
    prob.model.connect('c_duct', ['climb.manta.ducts.l_char','cruise.ducts.l_char'])
    prob.model.connect('l_char_ray', 'climb.ray.canard.l_char')

    prob.model.connect('l_pod', ['climb.ray.pods.l_char','climb.ray.l_pod'])
    prob.model.connect('d_pod', ['climb.ray.d_pod'])

    prob.model.connect('k_WL_manta', ['climb.manta.k_WL','cruise.k_WL'])
    prob.model.connect('k_WL_ray', 'climb.ray.k_WL')
    prob.model.connect('h_winglet_manta', ['climb.manta.h_winglet','cruise.h_winglet'])
    prob.model.connect('h_winglet_ray', 'climb.ray.h_winglet')

    prob.model.connect('span_manta', ['climb.manta.span','cruise.span','manta_weight.b_w'])
    prob.model.connect('span_ray', ['climb.ray.span','ray_weight.b_w'])
    #prob.model.connect('l_char_manta', 'climb.manta.wing.l_char')
    #prob.model.connect('l_char_ray', 'climb.ray.canard.l_char')


    #prob.model.connect('ducted_fan_eta', 'climb.manta.ducts.eta')

    prob.model.connect('eta_motor_ray', 'climb.eta_motor_ray')
    prob.model.connect('eta_motor_manta', 'climb.eta_motor_manta')
    prob.model.connect('eta_inverter_ray', 'climb.eta_inverter_ray')
    prob.model.connect('eta_inverter_manta', 'climb.eta_inverter_manta')


    prob.model.connect('eta_cable_ray', 'climb.eta_cable_ray')
    prob.model.connect('eta_cable_manta', 'climb.eta_cable_manta')
    prob.model.connect('eta_battery', 'climb.eta_battery')
    prob.model.connect('eta_generator', ['climb.eta_generator','cruise.eta_gen'])



    prob.model.connect('eta_gen_inverter', 'climb.eta_gen_inverter')
    prob.model.connect('psfc', 'climb.psfc')

    prob.model.connect('eta_fan', 'cruise.eta_fan')
    prob.model.connect('eta_duct', 'cruise.eta_duct')
    

    # Propulsion system parameters
    prob.model.connect('eta_pe', 'cruise.eta_pe')
    prob.model.connect('eta_motor', 'cruise.eta_motor')


    prob.model.connect('k_lam_manta', ['climb.manta.wing.k_lam','cruise.wing.k_lam'])
    prob.model.connect('k_lam_duct', ['climb.manta.ducts.k_lam','cruise.ducts.k_lam'])
    prob.model.connect('k_lam_ray', 'climb.ray.canard.k_lam')
    prob.model.connect('k_lam_pod', 'climb.ray.pods.k_lam')



    prob.model.connect('sweep_max_t_manta', ['climb.manta.sweep_max_t','cruise.sweep_max_t'])
    prob.model.connect('sweep_max_t_ray', 'climb.ray.sweep_max_t')



    prob.model.connect('t_c_manta', ['climb.manta.t_c','cruise.t_c','manta_weight.t_c_w'])
    prob.model.connect('t_c_ray', ['climb.ray.t_c','ray_weight.t_c_w'])
    prob.model.connect('tau_manta', ['climb.manta.tau','cruise.tau'])
    prob.model.connect('tau_ray', 'climb.ray.tau')    


    prob.model.connect('ray_energy_ratio', 'climb.ray_energy_ratio')
    prob.model.connect('lambda_manta', ['climb.manta.lambda','cruise.lambda','manta_weight.lambda'])
    prob.model.connect('lambda_ray', ['climb.ray.lambda','ray_weight.lambda'])



    prob.model.connect('num_ducts', ['climb.manta.num_ducts','cruise.num_ducts','n_motors_manta'])
    prob.model.connect('num_pods', ['climb.ray.num_pods','n_motors_ray'])

    prob.model.connect('od_duct', ['climb.manta.od_duct','cruise.od_duct'])
    prob.model.connect('id_duct', ['climb.manta.id_duct','cruise.id_duct'])
    prob.model.connect('c_duct', ['climb.manta.c_duct','cruise.c_duct'])



    #prob.model.connect('n_prplsrs', 'propeller.n_motors')
    prob.model.connect('k_prplsr', ['ray_weight.prop_weight.k_prplsr','manta_weight.compute_fan_weight.k_prplsr'])


    model = prob.model
    # Setup optimization
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'IPOPT'
    
    # Define design variable
    model.add_design_var('mto_manta', lower=8600, upper=99999, units='kg',ref = 100000)
    model.add_design_var('ray_mass_ratio', lower=1e-3, upper=0.99)
    model.add_design_var('duration', lower=100.0, upper=59.5*60.0, units='s',ref=3600)
    
    #model.add_constraint('ray_battery_mass_fraction.fraction_margin', lower=1e-3)
    #model.add_constraint('manta_battery_mass_fraction.fraction_margin', lower=1e-3)
    model.add_constraint('climb.z1_margin', upper  = 30)
    model.add_constraint('mantaray_mtom_margin.mtom_margin', upper=100)

    # Define objective
    model.add_objective('cruise.range',scaler = -1)

    
    # Setup recorder
    recorder = om.SqliteRecorder('toc_cases.sql')
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['includes'] = ['*']
    prob.driver.recording_options['record_objectives'] = True
    prob.driver.recording_options['record_desvars'] = True


    # Setup problem
    prob.setup()
    #om.n2(prob)

    # Run model
    prob.run_model()

    prob.run_driver()


    
    print("\n=== Key Results ===")
    print(f"Manta-Ray System MTOM: {prob['sum_MTOW.manta_ray_mto_calc'][0]/9.806:.0f} kg")

    print(f"Manta MTOM: {prob['manta_weight.w_mto_calc'][0]/9.806:.0f} kg")
    print(f"Ray MTOM: {prob['ray_weight.mto_calc'][0]/9.806:.0f} kg")
    print("\nPower Requirements:")

    print(f"Manta Unit Power: {prob['p_shaft_manta_climb'][0]/1000:.0f} kW")
    print(f"Ray Unit Power: {prob['p_shaft_ray_climb'][0]/1000:.0f} kW")
    print(f"\nRange: {prob['cruise.range'][0]/1000:.0f} km")

    print(f"Duration (min): {prob['duration'][0]/60:.0f}")



    print("\nBattery Mass Fractions:")
    print(f"Manta: {(1 - prob['manta_battery_mass_fraction.fraction_margin']):.3f}")
    print(f"Ray: {(1 - prob['ray_battery_mass_fraction.fraction_margin']):.3f}")




    
    # Print results

    
    # Check partials
    #prob.check_partials(compact_print=True) 