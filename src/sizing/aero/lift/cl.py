import openmdao.api as om
import numpy as np

class LiftCoefficient(om.ExplicitComponent):
    """
    Computes lift coefficient using:
    CL = CL0 + CL_alpha_eff * alpha
    
    Inputs:
        CL0 : float
            Zero-angle lift coefficient [-]
        CL_alpha_eff : float
            Effective lift curve slope [1/rad]
        alpha : float
            Angle of attack [rad]
    
    Outputs:
        CL : float
            Lift coefficient [-]
    """

    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')
    
    def setup(self):
        N = self.options['N']

        self.add_input('CL0', val=1.0 * np.ones(N), desc='Zero-angle lift coefficient')

        self.add_input('CL_alpha_eff', val=1.0 * np.ones(N), units='1/rad',
                      desc='Effective lift curve slope')
        self.add_input('alpha', val=1.0 * np.ones(N), units='rad',
                      desc='Angle of attack')



        self.add_output('CL', val=1.0 * np.ones(N), desc='Lift coefficient')
        
        self.declare_partials('CL', ['CL0', 'CL_alpha_eff', 'alpha'])
        
    def compute(self, inputs, outputs):
        outputs['CL'] = inputs['CL0'] + inputs['CL_alpha_eff'] * inputs['alpha']
        
    def compute_partials(self, inputs, partials):

        N = self.options['N']
        
        partials['CL', 'CL0'] = 1
        partials['CL', 'CL_alpha_eff'] = np.eye(N)*inputs['alpha']
        partials['CL', 'alpha'] = np.eye(N)*inputs['CL_alpha_eff']
