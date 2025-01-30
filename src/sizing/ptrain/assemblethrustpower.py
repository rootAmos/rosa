import openmdao.api as om
from .computethrustdistribution import ThrustDistribution
from .computeductedfan import ComputeDuctedFan
from .computepropeller import CruisePropeller

class ThrustPowerGroup(om.Group):
    """
    Group that computes power requirements from thrust distribution.
    """
    
    def setup(self):
        # Add thrust distribution
        self.add_subsystem('thrust_dist', ThrustDistribution(),
                          promotes_inputs=['thrust_required', 'thrust_ratio',
                                         'num_ray', 'num_manta'])
        
        # Add propulsion components
        self.add_subsystem('ductedfan', ComputeDuctedFan(),
                          promotes_inputs=['rho', 'V'])
        self.add_subsystem('propeller', CruisePropeller(),
                          promotes_inputs=['rho', 'V'])
        
        # Connect thrust to propulsion components
        self.connect('thrust_dist.ray_thrust_unit', 'propeller.thrust')
        self.connect('thrust_dist.manta_thrust_unit', 'ductedfan.thrust') 