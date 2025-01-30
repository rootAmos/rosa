import openmdao.api as om
import numpy as np

class ThrustDistribution(om.ExplicitComponent):
    """
    Distributes total thrust between Ray and Manta based on thrust ratio.
    
    Inputs:
        thrust_required : float
            Total thrust required [N]
        thrust_ratio : float
            Ratio of Ray thrust to total thrust [-]
        num_ray : float
            Number of Ray motors [-]
        num_manta : float
            Number of Manta motors [-]
            
    Outputs:
        ray_thrust_total : float
            Total thrust allocated to Ray [N]
        manta_thrust_total : float
            Total thrust allocated to Manta [N]
        ray_thrust_unit : float
            Thrust per Ray motor [N]
        manta_thrust_unit : float
            Thrust per Manta motor [N]
    """
    
    def setup(self):
        # Inputs
        self.add_input('thrust_required', val=0.0, units='N',
                      desc='Total thrust required')
        self.add_input('thrust_ratio', val=0.2,
                      desc='Ray thrust / total thrust')
        self.add_input('num_ray', val=1.0,
                      desc='Number of Ray motors')
        self.add_input('num_manta', val=1.0,
                      desc='Number of Manta motors')
        
        # Outputs
        self.add_output('ray_thrust_total', units='N',
                       desc='Total Ray thrust')
        self.add_output('manta_thrust_total', units='N',
                       desc='Total Manta thrust')
        self.add_output('ray_thrust_unit', units='N',
                       desc='Thrust per Ray motor')
        self.add_output('manta_thrust_unit', units='N',
                       desc='Thrust per Manta motor')
        
        self.declare_partials('*', '*')
        
    def compute(self, inputs, outputs):
        thrust_req = inputs['thrust_required']
        thrust_ratio = inputs['thrust_ratio']
        
        # Compute total thrusts
        outputs['ray_thrust_total'] = thrust_req * thrust_ratio
        outputs['manta_thrust_total'] = thrust_req * (1.0 - thrust_ratio)
        
        # Compute unit thrusts
        outputs['ray_thrust_unit'] = outputs['ray_thrust_total'] / inputs['num_ray']
        outputs['manta_thrust_unit'] = outputs['manta_thrust_total'] / inputs['num_manta']
        
    def compute_partials(self, inputs, partials):
        thrust_req = inputs['thrust_required']
        thrust_ratio = inputs['thrust_ratio']
        num_ray = inputs['num_ray']
        num_manta = inputs['num_manta']
        
        # Ray total thrust partials
        partials['ray_thrust_total', 'thrust_required'] = thrust_ratio
        partials['ray_thrust_total', 'thrust_ratio'] = thrust_req
        partials['ray_thrust_total', 'num_ray'] = 0.0
        partials['ray_thrust_total', 'num_manta'] = 0.0
        
        # Manta total thrust partials
        partials['manta_thrust_total', 'thrust_required'] = 1.0 - thrust_ratio
        partials['manta_thrust_total', 'thrust_ratio'] = -thrust_req
        partials['manta_thrust_total', 'num_ray'] = 0.0
        partials['manta_thrust_total', 'num_manta'] = 0.0
        
        # Ray unit thrust partials
        partials['ray_thrust_unit', 'thrust_required'] = thrust_ratio / num_ray
        partials['ray_thrust_unit', 'thrust_ratio'] = thrust_req / num_ray
        partials['ray_thrust_unit', 'num_ray'] = -thrust_req * thrust_ratio / (num_ray**2)
        partials['ray_thrust_unit', 'num_manta'] = 0.0
        
        # Manta unit thrust partials
        partials['manta_thrust_unit', 'thrust_required'] = (1.0 - thrust_ratio) / num_manta
        partials['manta_thrust_unit', 'thrust_ratio'] = -thrust_req / num_manta
        partials['manta_thrust_unit', 'num_ray'] = 0.0
        partials['manta_thrust_unit', 'num_manta'] = -thrust_req * (1.0 - thrust_ratio) / (num_manta**2) 