import openmdao.api as om
import os
import sys
import pdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.sizing.ptrain.assemblethrustpower import ThrustPowerGroup
from src.sizing.mission.group_climb import ClimbProfile
from src.sizing.mission.atmos import ComputeAtmos
from src.sizing.aero.lift.group_lift_mantaray import GroupLiftMantaRay
from src.sizing.aero.drag.group_drag import GroupDragMantaRay
from src.sizing.mission.computethrustrequired import ThrustRequired




class TopOfClimbReq(om.Group):
    """Group for computing climb phase power and time requirements"""
    
    def initialize(self):
        self.options.declare('num_nodes', default=50)
        self.options.declare('manta', default=1)
        self.options.declare('ray', default=1)
        
    def setup(self):
        nn = self.options['num_nodes']
        
        # Add atmosphere computation
        self.add_subsystem('atmos', 
                          ComputeAtmos(),
                          promotes_inputs=['alt'],
                          promotes_outputs=['rho', 'temp', 'pressure'])
        
        # Add climb trajectory computation
        self.add_subsystem('climb',
                          ClimbProfile(num_nodes=nn),
                          promotes_inputs=['V_eas', 'gamma', 'h_start', 'h_end'])
        
        # Add lift computation
        self.add_subsystem('lift',
                          GroupLift(manta=self.options['manta'], 
                                  ray=self.options['ray']),
                          promotes_inputs=['rho', 'u'])
        
        # Add drag computation
        self.add_subsystem('drag',
                          GroupDragMantaRay(manta=self.options['manta'],
                                          ray=self.options['ray']),
                          promotes_inputs=['rho', 'u'])
        
        # Add thrust requirement computation
        self.add_subsystem('thrust_req',
                          ThrustReq(),
                          promotes_inputs=['w_mto'])
        
        # Add thrust power computation
        self.add_subsystem('thrust_power',
                          ThrustPowerGroup(),
                          promotes_inputs=['rho', 'thrust_ratio', 'num_ray', 'num_manta'])
        
        # Connect climb altitude to atmosphere
        self.connect('climb.alt', 'atmos.alt')
        
        # Connect true airspeed to lift/drag computations
        self.connect('climb.V_tas', 'u')
        self.connect('climb.V_tas', 'thrust_power.V')
        
        # Connect to thrust requirement
        self.connect('drag.D_total', 'thrust_req.drag')
        self.connect('climb.gamma_rad', 'thrust_req.gamma')
        self.connect('climb.udot', 'thrust_req.udot')
        
        # Connect thrust to power calculation
        self.connect('thrust_req.thrust_total', 'thrust_power.thrust_required')

if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Add group
    prob.model = TopOfClimbReq(num_nodes=50)
    
    # Setup problem
    prob.setup()
    
    # Set inputs
    prob.set_val('h_start', 0.0, units='m')
    prob.set_val('h_end', 10000.0, units='m')
    prob.set_val('V_eas', 250.0, units='kn')
    prob.set_val('gamma', 5.0, units='deg')
    prob.set_val('w_mto', 100000.0, units='N')
    
    # Set thrust power group inputs
    prob.set_val('thrust_ratio', 0.7)  # Ratio between ray and manta thrust
    prob.set_val('num_ray', 4)         # Number of ray propulsors
    prob.set_val('num_manta', 2)       # Number of manta propulsors
    
    # Run model
    prob.run_model()
    
    # Print results
    print("\nClimb Profile Results:")
    print(f"Duration: {prob.get_val('climb.duration')[0]:.1f} seconds")
    print(f"Required thrust: {prob.get_val('thrust_req.thrust_total')[0]:.1f} N")
    
    # Get power values for plotting
    ray_power = prob.get_val('thrust_power.ductedfan.power')
    manta_power = prob.get_val('thrust_power.propeller.power')
    total_power = ray_power + manta_power
    
    # Plot results
    import matplotlib.pyplot as plt
    
    time = prob.get_val('climb.time')
    alt = prob.get_val('climb.alt')
    V_tas = prob.get_val('climb.V_tas')
    thrust = prob.get_val('thrust_req.thrust_total')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Altitude vs Time
    ax1.plot(time, alt)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)')
    ax1.grid(True)
    ax1.set_title('Climb Profile')
    
    # True Airspeed vs Time
    ax2.plot(time, V_tas)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('True Airspeed (m/s)')
    ax2.grid(True)
    ax2.set_title('Speed Profile')
    
    # Power vs Time
    ax3.plot(time, ray_power/1000, label='Ray')
    ax3.plot(time, manta_power/1000, label='Manta')
    ax3.plot(time, total_power/1000, 'k--', label='Total')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Power (kW)')
    ax3.grid(True)
    ax3.legend()
    ax3.set_title('Power Requirements')
    
    # Thrust vs Time
    ax4.plot(time, thrust/1000)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Required Thrust (kN)')
    ax4.grid(True)
    ax4.set_title('Thrust Profile')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate energy used during climb
    energy_used = np.trapz(total_power, time)
    print(f"\nTotal energy used in climb: {energy_used/3.6e6:.2f} kWh")