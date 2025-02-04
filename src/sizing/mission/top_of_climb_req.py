import openmdao.api as om
import os
import sys
import pdb
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.sizing.ptrain.thrust2power import ThrustPowerGroup
from src.sizing.mission.group_climb import ClimbGroup
from src.sizing.aero.lift.group_cl_mantaray import GroupCLMantaRay
from src.sizing.aero.drag.drag_force import DragForce
from src.sizing.mission.thrust_req import ThrustReq
from src.sizing.aero.drag.group_cd_mantaray import GroupCDMantaRay

from src.sizing.ptrain.power_conso_mantaray import PowerConsoGroup
from src.sizing.mission.mach_number import MachNumber



class TopOfClimbReq(om.Group):
    """Group for computing climb phase power and time requirements"""
    
    def initialize(self):
        self.options.declare('N', default=1)
        self.options.declare('manta', default=1)
        self.options.declare('ray', default=1)
        
    def setup(self):
        N = self.options['N']
        
        
        # Add climb trajectory computation
        self.add_subsystem('climb',
                          ClimbGroup(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
        
        self.add_subsystem('mach',
                          MachNumber(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
        
        # Add lift computation
        self.add_subsystem('lift',

                          GroupCLMantaRay(manta=self.options['manta'], 
                                  ray=self.options['ray'], N=N),
                          promotes_inputs=['*'],

                          promotes_outputs=['*'])
        

        self.add_subsystem('cd',
                          GroupCDMantaRay(manta=self.options['manta'], 
                                  ray=self.options['ray'], N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

        
        self.add_subsystem('drag',
                          DragForce(N=N),
                          promotes_inputs=['rho', 'u','S_ref'],
                          promotes_outputs=['*'])

        
        self.add_subsystem('thrust_req',
                          ThrustReq(N=N),
                          promotes_inputs=['*'], 
                          promotes_outputs=['*'])
        

        self.add_subsystem('thrust_power',
                          ThrustPowerGroup(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
        
        self.add_subsystem('power_conso', PowerConsoGroup(N=N), 
                    promotes_inputs=['*'],
                    promotes_outputs=['*'])

        
        self.connect('CD_manta_ray', 'drag.CD')
        self.connect('mach', ['cl_alpha_manta.mach','cl_alpha_ray.mach'])
        #self.connect('alpha',['manta.alpha','ray.alpha'])

        self.connect('cl_manta.CL', 'manta.CL')
        self.connect('cl_ray.CL', 'ray.CL')

        self.connect('manta_cl0.CL0', 'manta.CL0')
        self.connect('ray_cl0.CL0', 'ray.CL0')

        self.connect('thrust_total','thrust_required')

        self.connect('ductedfan.p_shaft_unit', 'm_shaft_power_unit')
        self.connect('propeller.p_shaft_unit', 'r_shaft_power_unit')






if __name__ == "__main__":


    # Create problem instance
    prob = om.Problem()
    
    # Independent variable component
    ivc = om.IndepVarComp()
    
    # Number of points
    N = 50
    
    # Initial conditions
    z = np.linspace(1500, 30000, N)
    
    # Add inputs to ivc

    # Mission parameters
    ivc.add_output('u_eas', val=70, units='m/s', desc='Equivalent airspeed')
    ivc.add_output('gamma', val=5.0 * np.ones(N), units='deg', desc='Climb angle')
    ivc.add_output('w_mto', val=100000.0, units='N', desc='Maximum takeoff weight')
    ivc.add_output('duration', val=600.0, units='s', desc='Climb duration')
    ivc.add_output('z0', val=1500, units='ft', desc='Initial altitude')
    ivc.add_output('z1', val=30000, units='ft', desc='Final altitude')
    ivc.add_output('mu', val=1.789e-5 * np.ones(N), units='Pa*s', desc='Dynamic viscosity')
    ivc.add_output('ray_energy_ratio', val=0.3)
    ivc.add_output('mto', val=10000.0, units='kg', desc='Maximum takeoff mass')

    # Thrust power group inputs
    ivc.add_output('thrust_ratio', val=0.7 * np.ones(N), desc='Ratio between ray and manta thrust')
    ivc.add_output('num_pods', val=6, desc='Number of ray propulsors')
    ivc.add_output('num_ducts', val=4, desc='Number of manta propulsors')

    # Ducted fan inputs
    ivc.add_output('d_blades_duct', 1.5, units='m')
    ivc.add_output('d_hub_duct', 0.5, units='m')
    ivc.add_output('epsilon_r', 1.5, units=None)
    ivc.add_output('eta_fan', 0.9 , units=None)
    ivc.add_output('eta_duct', 0.5, units=None)
    ivc.add_output('c_duct', val=3.0, units='m', desc='Nacelle length')
    ivc.add_output('od_duct', val=2.0, units='m', desc='Duct outer diameter')
    ivc.add_output('id_duct', val=1.0, units='m', desc='Duct inner diameter')
    ivc.add_output('Q_duct', val=1.0, desc='Duct interference factor')
    ivc.add_output('k_lam_duct', val=0.1, desc='Duct laminar flow fraction')
    ivc.add_output('num_ducts', val=2, desc='Number of ducts')

    # Propeller inputs
    ivc.add_output('d_blades_prop', 1.5, units='m')
    ivc.add_output('d_hub_prop', 0.5, units='m')
    ivc.add_output('eta_prop', 0.9 , units=None)
    ivc.add_output('l_pod', val=3.0, units='m', desc='Pod length')
    ivc.add_output('d_pod', val=2.0, units='m', desc='Pod diameter')
    ivc.add_output('Q_pod', val=1.0, desc='Pod interference factor')
    ivc.add_output('k_lam_pod', val=0.1, desc='Pod laminar flow fraction')



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
    ivc.add_output('phi_50_manta', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil_manta', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')
    ivc.add_output('aspect_ratio_manta', val=10.0, desc='Manta aspect ratio')
    ivc.add_output('d_eps_manta_d_alpha', val=0.25, desc='Manta downwash derivative')
    ivc.add_output('alpha0_airfoil_manta', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_twist_manta', val=0.0, units='rad', desc='twist angle')
    ivc.add_output('sweep_25_manta', val=12.0, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('Q_manta', val=1.1, desc='Interference factor')
    ivc.add_output('S_exp_manta', val=50.0, units='m**2', desc='Exposed planform area')
    ivc.add_output('t_c_manta', val=0.19, desc='Thickness to chord ratio')
    ivc.add_output('tau_manta', val=0.8, desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
    ivc.add_output('lambda_manta', val=0.45, desc='Wing taper ratio')
    ivc.add_output('k_lam_manta', val=0.1, desc='Laminar flow fraction')
    ivc.add_output('sweep_max_t_manta', val=10,units='deg', desc='Wing sweep at maximum thickness')
    ivc.add_output('l_char_manta', val=1.0, units='m',desc='Characteristic length')
    ivc.add_output('h_winglet_manta', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('span_manta', val=30.0, units='m', desc='Wing span')
    ivc.add_output('k_WL_manta', val=2.83, desc='Winglet effectiveness factor')
    ivc.add_output('S_manta', val=100.0, units='m**2', desc='Manta reference area')


    # Ray Aero Parameters
    ivc.add_output('S_ray', val=100.0, units='m**2', desc='Ray area')
    ivc.add_output('phi_50_ray', val=5.0, units='deg', desc='Ray 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil_ray', val=2*np.pi, units='1/rad', desc='Ray airfoil lift curve slope')
    ivc.add_output('aspect_ratio_ray', val=10.0, desc='Ray aspect ratio')
    ivc.add_output('d_eps_ray_d_alpha', val=0.25, desc='Ray downwash derivative')
    ivc.add_output('alpha0_airfoil_ray', val=-2.0, units='deg', desc='Ray airfoil zero-lift angle')
    ivc.add_output('d_twist_ray', val=0.0, units='rad', desc='twist angle')
    ivc.add_output('sweep_25_ray', val=12.0, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('Q_ray', val=1.1, desc='Interference factor')
    ivc.add_output('S_exp_ray', val=50.0, units='m**2', desc='Exposed planform area')
    ivc.add_output('t_c_ray', val=0.19, desc='Thickness to chord ratio')
    ivc.add_output('tau_ray', val=0.8, desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
    ivc.add_output('lambda_ray', val=0.45, desc='Wing taper ratio')
    ivc.add_output('k_lam_ray', val=0.1, desc='Laminar flow fraction')
    ivc.add_output('sweep_max_t_ray', val=10,units='deg', desc='Wing sweep at maximum thickness')
    ivc.add_output('l_char_ray', val=1.0, units='m',desc='Characteristic length')
    ivc.add_output('h_winglet_ray', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('span_ray', val=30.0, units='m', desc='Wing span')
    ivc.add_output('k_WL_ray', val=2.83, desc='Winglet effectiveness factor')

    manta_opt = 1
    ray_opt = 1

    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes=['*'])
    model.add_subsystem('toc', TopOfClimbReq(N=N,manta=1,ray=1), promotes=['*'])

    # Duted Fan parameters
    prob.model.connect('d_blades_duct', 'ductedfan.d_blades')
    prob.model.connect('d_hub_duct', 'ductedfan.d_hub')
    prob.model.connect('d_blades_prop', 'propeller.d_blades')
    prob.model.connect('d_hub_prop', 'propeller.d_hub')
    prob.model.connect('eta_prop', 'propeller.eta_prop')
    prob.model.connect('eta_fan', 'ductedfan.eta_fan')
    prob.model.connect('eta_duct', 'ductedfan.eta_duct')
    prob.model.connect('num_pods', 'propeller.n_motors')
    prob.model.connect('num_ducts', 'ductedfan.num_ducts')
    prob.model.connect('epsilon_r', 'ductedfan.epsilon_r')
    prob.model.connect('l_char_manta', 'manta.wing.l_char')
    prob.model.connect('c_duct', 'manta.ducts.l_char')
    prob.model.connect('l_char_ray', 'ray.canard.l_char')
    prob.model.connect('k_lam_duct', 'manta.ducts.k_lam')
    prob.model.connect('Q_duct', 'manta.ducts.Q')
    prob.model.connect('num_ducts', 'manta.num_ducts')
    prob.model.connect('od_duct', 'manta.od_duct')
    prob.model.connect('id_duct', 'manta.id_duct')
    prob.model.connect('c_duct', 'manta.c_duct')

    # Propeller parameters
    prob.model.connect('l_pod', ['ray.pods.l_char','ray.l_pod'])
    prob.model.connect('d_pod', ['ray.d_pod'])
    prob.model.connect('Q_pod', 'ray.pods.Q')
    prob.model.connect('k_lam_pod', 'ray.pods.k_lam')
    prob.model.connect('num_pods', 'ray.num_pods')


    # Manta Aero  parameters
    prob.model.connect('aspect_ratio_manta', 'cl_alpha_manta.aspect_ratio')  
    prob.model.connect('aspect_ratio_ray', 'cl_alpha_ray.aspect_ratio')
    prob.model.connect('k_lam_manta', 'manta.wing.k_lam')
    prob.model.connect('Q_manta', 'manta.wing.Q')
    prob.model.connect('sweep_max_t_manta', 'manta.sweep_max_t')
    prob.model.connect('t_c_manta', 'manta.t_c')
    prob.model.connect('tau_manta', 'manta.tau')
    prob.model.connect('lambda_manta', 'manta.lambda')
    prob.model.connect('alpha0_airfoil_manta', 'manta_cl0.alpha0_airfoil')
    prob.model.connect('phi_50_manta', ['cl_alpha_manta.phi_50'])
    prob.model.connect('d_twist_manta', ['manta_cl0.d_twist'])
    prob.model.connect('cl_alpha_airfoil_manta', ['cl_alpha_manta.cl_alpha_airfoil'])
    prob.model.connect('aspect_ratio_manta', 'manta.aspect_ratio')
    prob.model.connect('sweep_25_manta', 'manta.sweep_25')
    prob.model.connect('h_winglet_manta', 'manta.h_winglet')
    prob.model.connect('k_WL_manta', 'manta.k_WL')
    prob.model.connect('span_manta', 'manta.span')
    prob.model.connect('S_manta', 'S_ref')
    prob.model.connect('S_exp_manta', 'manta.S_exp')


    # Ray Aero parameters
    prob.model.connect('k_lam_ray', 'ray.canard.k_lam')
    prob.model.connect('Q_ray', 'ray.canard.Q')
    prob.model.connect('sweep_max_t_ray', 'ray.sweep_max_t')
    prob.model.connect('t_c_ray', 'ray.t_c')
    prob.model.connect('tau_ray', 'ray.tau')    
    prob.model.connect('lambda_ray', 'ray.lambda')
    prob.model.connect('alpha0_airfoil_ray', 'ray_cl0.alpha0_airfoil')
    prob.model.connect('phi_50_ray', ['cl_alpha_ray.phi_50'])
    prob.model.connect('d_twist_ray', ['ray_cl0.d_twist'])
    prob.model.connect('cl_alpha_airfoil_ray', ['cl_alpha_ray.cl_alpha_airfoil'])
    prob.model.connect('aspect_ratio_ray', 'ray.aspect_ratio')
    prob.model.connect('sweep_25_ray', 'ray.sweep_25')
    prob.model.connect('h_winglet_ray', 'ray.h_winglet')
    prob.model.connect('k_WL_ray', 'ray.k_WL')
    prob.model.connect('span_ray', 'ray.span')
    prob.model.connect('S_exp_ray', 'ray.S_exp')

    
    # Setup optimization
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'IPOPT'
    
    # Define design variable
    model.add_design_var('duration', lower=100.0, upper=2000.0, units='s')
    
    # Define objective
    model.add_objective('z1_margin')

    
    # Setup recorder
    recorder = om.SqliteRecorder('toc_cases.sql')
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['includes'] = ['*']
    prob.driver.recording_options['record_objectives'] = True
    prob.driver.recording_options['record_desvars'] = True
    
    # Setup problem
    prob.setup()
    #om.n2(prob)
    
    # Run optimization
    prob.run_driver()
    
    # Print results
    print("\nTop of Climb Results:")
    print(f"Duration: {prob.get_val('duration')[0]/60:.1f} minutes")
    print(f"Altitude margin: {prob.get_val('z1_margin')[0]:.1f} meters")
    print(f"\nPower Requirements:")
    print(f"Max Ray  Unit power: {max(prob.get_val('ductedfan.p_shaft_unit'))/1000:.1f} kW")
    print(f"Max Manta Unit power: {max(prob.get_val('propeller.p_shaft_unit'))/1000:.1f} kW")
    print(f"\nEnergy Requirements:")
    print(f"Ray Battery Energy: {prob.get_val('ray_battery_energy')[0]/3.6e6:.1f} kWh")
    print(f"Manta Fuel Consumption: {prob.get_val('manta_fuel_mass')[0]:.1f} kg")
    


    import pdb
    pdb.set_trace()
    # Plot results
    import matplotlib.pyplot as plt
    
    time = prob.get_val('time')
    z = prob.get_val('z') * 3.28084  # Convert to feet
    u = prob.get_val('u')
    zdot = prob.get_val('zdot') * 196.85  # Convert to ft/min
    udot = prob.get_val('udot')
    ray_power = prob.get_val('ray_elec_power') /1000  # Convert to kW
    manta_power = prob.get_val('manta_elec_power')/1000  # Convert to kW
    total_power = ray_power + manta_power
    


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.plot(z, u)
    ax1.set_xlabel('Altitude (ft)')
    ax1.set_ylabel('True Airspeed (m/s)')
    ax1.grid(True)
    ax1.set_title('Speed Profile')
    
    ax2.plot(z, udot)
    ax2.set_xlabel('Altitude (ft)')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.grid(True)
    ax2.set_title('Acceleration Profile')
    
    ax3.plot(z, zdot)
    ax3.set_xlabel('Altitude (ft)')
    ax3.set_ylabel('Climb Rate (ft/min)')
    ax3.grid(True)
    ax3.set_title('Climb Rate Profile')
    
    ax4.plot(z, ray_power, label='Ray')
    ax4.plot(z, manta_power, label='Manta')
    ax4.plot(z, total_power, 'k--', label='Total')
    ax4.set_xlabel('Altitude (ft)')
    ax4.set_ylabel('Power (kW)')
    ax4.grid(True)
    ax4.legend()
    ax4.set_title('Power Requirements')
    
    plt.tight_layout()
    plt.show()