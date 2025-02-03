import openmdao.api as om
import numpy as np

from src.sizing.mission.top_of_climb_req import TopOfClimbReq
from src.sizing.mission.turbo_cruise import ComputeTurboCruiseRange, PropulsiveEfficiency
from src.sizing.ptrain.power_conso_mantaray import PowerConsumptionGroup

class MissionMantaRay(om.Group):
    """
    Group that simulates complete mission profile:
    1. Climb phase with both Manta and Ray (Ray provides electrical power)
    2. Cruise phase with only Manta (using turbogenerator)
    """
    
    def initialize(self):
        self.options.declare('N_climb', default=50, desc='Number of nodes in climb')
    
    def setup(self):
        N_climb = self.options['N_climb']

        # Add climb phase analysis
        self.add_subsystem('climb',
                          TopOfClimbReq(N=N_climb),
                          promotes_inputs=['*'],
                          promotes_outputs=['z1'])

        # Add propulsive efficiency calculation
        self.add_subsystem('fan_eff',
                          PropulsiveEfficiency(),
                          promotes_inputs=['eta_fan', 'eta_duct', 'eta_prplsv'],
                          promotes_outputs=['eta_pt'])

        # Add cruise range calculation
        self.add_subsystem('cruise',
                          ComputeTurboCruiseRange(),
                          promotes_inputs=['cl', 'cd', 'eta_pe', 'eta_motor', 
                                        'eta_pt', 'eta_gen', 'cp', 'w_mto',
                                        'w_fuel', 'target_range'],
                          promotes_outputs=['range'])

        # Set climb phase to use Ray's battery (ray_energy_ratio = 1.0)
        self.set_input_defaults('climb_power.ray_energy_ratio', val=1.0)

        # Connect climb outputs to cruise inputs
        self.connect('CL', 'cl')
        self.connect('CD', 'cd')
        self.connect('climb.power_required', 'climb_power.total_elec_power')
        self.connect('mto', ['w_mto', 'climb.mto'])


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Independent variable component
    ivc = om.IndepVarComp()
    
    N_climb = 50
    
    # Add climb phase inputs
    ivc.add_output('z0', val=0.0, units='m')
    ivc.add_output('z1', val=10000.0, units='m')
    ivc.add_output('u_eas', val=100.0, units='m/s')
    ivc.add_output('gamma', val=5.0, units='deg')
    
    # Add aircraft parameters
    ivc.add_output('mto', val=10000.0, units='kg')
    ivc.add_output('S_ref', val=100.0, units='m**2')
    ivc.add_output('S_ray', val=20.0, units='m**2')
    
    # Add aerodynamic parameters
    ivc.add_output('phi_50_manta', val=5.0, units='deg')
    ivc.add_output('phi_50_ray', val=5.0, units='deg')
    ivc.add_output('cl_alpha_airfoil_manta', val=2*np.pi, units='1/rad')
    ivc.add_output('cl_alpha_airfoil_ray', val=2*np.pi, units='1/rad')
    ivc.add_output('aspect_ratio_manta', val=10.0)
    ivc.add_output('aspect_ratio_ray', val=8.0)
    ivc.add_output('alpha0_airfoil_manta', val=-2.0, units='deg')
    ivc.add_output('alpha0_airfoil_ray', val=-2.0, units='deg')
    ivc.add_output('d_twist_manta', val=0.0, units='rad')
    ivc.add_output('d_twist_ray', val=0.0, units='rad')
    
    # Add propulsion parameters
    ivc.add_output('eta_fan', val=0.95)
    ivc.add_output('eta_duct', val=0.96)
    ivc.add_output('eta_prplsv', val=0.98)
    ivc.add_output('eta_pe', val=0.95)
    ivc.add_output('eta_motor', val=0.95)
    ivc.add_output('eta_gen', val=0.95)
    ivc.add_output('eta_battery', val=0.95)
    ivc.add_output('eta_generator', val=0.95)
    ivc.add_output('eta_gen_inverter', val=0.95)
    
    # Add mission parameters
    ivc.add_output('target_range', val=1000.0, units='km')
    ivc.add_output('w_fuel', val=500.0*9.81, units='N')
    ivc.add_output('cp', val=0.5/60/60/1000, units='kg/W/s')
    ivc.add_output('psfc', val=1.7e-5, units='kg/W/s')
    
    # Build the model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('mission', 
                            MissionMantaRay(N_climb=N_climb),
                            promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run model
    prob.run_model()
    
    # Print results
    print("\nMission Results:")
    print("---------------")
    print("Climb Phase:")
    print(f"CL: {prob.get_val('CL')[0]:.3f}")
    print(f"CD: {prob.get_val('CD')[0]:.3f}")
    print(f"Ray battery energy: {prob.get_val('ray_battery_energy')[0]/3.6e6:.1f} kWh")
    
    print("\nCruise Phase:")
    print(f"Range: {prob.get_val('range')[0]/1000:.1f} km")
    print(f"Total propulsive efficiency: {prob.get_val('eta_pt')[0]:.3f}")
    
    # Check partials
    prob.check_partials(compact_print=True) 