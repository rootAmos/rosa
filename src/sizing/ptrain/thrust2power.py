import openmdao.api as om
import numpy as np
import os
import pdb
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.sizing.ptrain.thrust_distribution import ThrustDistribution
from src.sizing.ptrain.ductedfan import ComputeDuctedFan
from src.sizing.ptrain.propeller import CruisePropeller
from src.sizing.ptrain.power_conso_mantaray import PowerConsoGroup

class ThrustPowerGroup(om.Group):
    """
    Group that computes power requirements from thrust distribution.
    """

    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')
    
    def setup(self):
        # Add thrust distribution

        N = self.options['N']

        self.add_subsystem('thrust_dist', ThrustDistribution(N=N),
                          promotes_inputs=['thrust_required', 'thrust_ratio',
                                         'num_pods', 'num_fans'])

        # Add propulsion components
        self.add_subsystem('ductedfan', ComputeDuctedFan(N=N),
                          promotes_inputs=['rho', 'u'])

        self.add_subsystem('propeller', CruisePropeller(N=N),
                          promotes_inputs=['rho', 'u'])
        


        # Connect thrust to propulsion components

        self.connect('thrust_dist.ray_thrust_total', 'propeller.thrust_total')
        self.connect('thrust_dist.manta_thrust_unit', 'ductedfan.thrust_unit') 


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Independent variable component
    ivc = om.IndepVarComp()
    
    N = 50  # number of nodes
    
    # Add inputs to ivc
    ivc.add_output('thrust_required', val=np.linspace(8600 * 9.806 /17, 8600 * 9.806 /18, N), units='N', desc='Total thrust required')
    ivc.add_output('thrust_ratio', val=0.5 * np.ones(N), desc='Ratio of thrust between ray and manta')
    ivc.add_output('num_pods', val=2, desc='Number of ray propulsors')
    ivc.add_output('num_fans', val=2, desc='Number of manta propulsors')
    ivc.add_output('rho', val=1.225 * np.ones(N), units='kg/m**3', desc='Air density')
    ivc.add_output('u', val=100.0 * np.ones(N), units='m/s', desc='Airspeed')

    # Ducted fan inputs
    ivc.add_output('d_blades_duct', 1.5, units='m')
    ivc.add_output('d_hub_duct', 0.5, units='m')
    ivc.add_output('epsilon_r', 1.5, units=None)
    ivc.add_output('eta_fan', 0.9 , units=None)
    ivc.add_output('eta_duct', 0.5, units=None)

    # Propeller inputs
    ivc.add_output('d_blades_prop', 1.5, units='m')
    ivc.add_output('d_hub_prop', 0.5, units='m')
    ivc.add_output('eta_prop', 0.9 , units=None)



    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes=['*'])
    model.add_subsystem('thrust_power', ThrustPowerGroup(N=N), promotes=['*'])


    prob.model.connect('d_blades_duct', 'ductedfan.d_blades')
    prob.model.connect('d_hub_duct', 'ductedfan.d_hub')
    prob.model.connect('d_blades_prop', 'propeller.d_blades')
    prob.model.connect('d_hub_prop', 'propeller.d_hub')

    prob.model.connect('eta_prop', 'propeller.eta_prop')
    prob.model.connect('eta_fan', 'ductedfan.eta_fan')
    prob.model.connect('eta_duct', 'ductedfan.eta_duct')
    prob.model.connect('num_pods', 'propeller.n_motors')
    prob.model.connect('num_fans', 'ductedfan.n_fans')
    prob.model.connect('epsilon_r', 'ductedfan.epsilon_r')
    





    # Setup problem
    prob.setup()



    om.n2(prob)
    
    # Run model
    prob.run_model()
    
    # Print results
    print("\nThrust-Power Results:")
    print(f"Total thrust required: {prob.get_val('thrust_required')[0]:.1f} N")
    print(f"Ray thrust per unit: {prob.get_val('thrust_dist.ray_thrust_unit')[0]:.1f} N")
    print(f"Manta thrust per unit: {prob.get_val('thrust_dist.manta_thrust_unit')[0]:.1f} N")
    print(f"Propeller power: {prob.get_val('propeller.p_shaft_unit')[0]:.1f} W")
    print(f"Ducted fan power: {prob.get_val('ductedfan.p_shaft_unit')[0]:.1f} W")
    

    # Plot results
    import matplotlib.pyplot as plt
    
    thrust_req = prob.get_val('thrust_required')
    prop_power = prob.get_val('propeller.p_shaft_unit')/1000
    fan_power = prob.get_val('ductedfan.p_shaft_unit')/1000
    #pdb.set_trace()
    



    plt.figure(figsize=(10, 6))
    plt.plot(thrust_req, prop_power, 'b-', label='Propeller Power')
    plt.plot(thrust_req, fan_power, 'r-', label='Ducted Fan Power')
    plt.plot(thrust_req, prop_power + fan_power, 'k--', label='Total Power')
    plt.xlabel('Thrust Required (N)')
    plt.ylabel('Power (kW)')
    plt.grid(True)
    plt.legend()
    plt.title('Power Requirements vs Thrust')
    plt.show()
    
    # Check partials
    # prob.check_partials(compact_print=True) 