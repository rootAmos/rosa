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
        num_pods : float
            Number of Ray motors [-]
        num_fans : float
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

    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')
    
    def setup(self):
        # Inputs

        N = self.options['N']

        self.add_input('thrust_required', val=1.0 * np.ones(N), units='N',
                      desc='Total thrust required')

        self.add_input('thrust_ratio', val=0.2 * np.ones(N),
                      desc='Ray thrust / total thrust')
        self.add_input('num_pods', val=1.0,
                      desc='Number of Ray motors')
        self.add_input('num_fans', val=1.0,
                      desc='Number of Manta motors')
        
        # Outputs
        self.add_output('ray_thrust_total', val=1.0 * np.ones(N), units='N',
                       desc='Total Ray thrust')
        self.add_output('manta_thrust_total', val=1.0 * np.ones(N), units='N',
                       desc='Total Manta thrust')

        self.add_output('ray_thrust_unit', val=1.0 * np.ones(N), units='N',
                       desc='Thrust per Ray motor')
        self.add_output('manta_thrust_unit', val=1.0 * np.ones(N), units='N',
                       desc='Thrust per Manta motor')
        
        self.declare_partials('*', '*')
        
    def compute(self, inputs, outputs):
        thrust_req = inputs['thrust_required']
        thrust_ratio = inputs['thrust_ratio']
        
        # Compute total thrusts
        outputs['ray_thrust_total'] = thrust_req * thrust_ratio
        outputs['manta_thrust_total'] = thrust_req * (1.0 - thrust_ratio)
        
        # Compute unit thrusts
        outputs['ray_thrust_unit'] = outputs['ray_thrust_total'] / inputs['num_pods']
        outputs['manta_thrust_unit'] = outputs['manta_thrust_total'] / inputs['num_fans']
        
    def compute_partials(self, inputs, partials):

        N = self.options['N']
        thrust_req = inputs['thrust_required']
        thrust_ratio = inputs['thrust_ratio']
        num_pods = inputs['num_pods']
        num_fans = inputs['num_fans']
        
        # Ray total thrust partials
        partials['ray_thrust_total', 'thrust_required'] = np.eye(N) * thrust_ratio
        partials['ray_thrust_total', 'thrust_ratio'] = np.eye(N) * thrust_req
        partials['ray_thrust_total', 'num_pods'] =  0.0
        partials['ray_thrust_total', 'num_fans'] = 0.0
        

        # Manta total thrust partials
        partials['manta_thrust_total', 'thrust_required'] = np.eye(N) * (1.0 - thrust_ratio)
        partials['manta_thrust_total', 'thrust_ratio'] = np.eye(N) * -thrust_req
        partials['manta_thrust_total', 'num_pods'] = 0.0
        partials['manta_thrust_total', 'num_fans'] = 0.0
        

        # Ray unit thrust partials
        partials['ray_thrust_unit', 'thrust_required'] = np.eye(N) * thrust_ratio / num_pods
        partials['ray_thrust_unit', 'thrust_ratio'] = np.eye(N) * thrust_req / num_pods
        partials['ray_thrust_unit', 'num_pods'] =  -thrust_req * thrust_ratio / (num_pods**2)

        partials['ray_thrust_unit', 'num_fans'] = 0.0
        
        # Manta unit thrust partials
        partials['manta_thrust_unit', 'thrust_required'] = np.eye(N) * (1.0 - thrust_ratio) / num_fans
        partials['manta_thrust_unit', 'thrust_ratio'] = np.eye(N) * -thrust_req / num_fans
        partials['manta_thrust_unit', 'num_pods'] = 0.0
        partials['manta_thrust_unit', 'num_fans'] = -thrust_req * (1.0 - thrust_ratio) / (num_fans**2) 