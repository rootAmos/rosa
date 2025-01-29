import openmdao.api as om
import numpy as np

class ZeroLiftAngle(om.ExplicitComponent):
    """
    Calculates the zero-lift angle of attack with Mach number corrections.
    
    Inputs:
        alpha_0 : float
            Basic zero-lift angle of attack at M=0.3 [rad]
        delta_alpha_0 : float
            Change in zero-lift angle due to flap deflection [rad]
        epsilon_t : float
            Twist angle [rad]
        alpha_0_M : float
            Zero-lift angle at current Mach number [rad]
        alpha_0_M03 : float
            Reference zero-lift angle at M=0.3 [rad]
    
    Outputs:
        alpha_0_W : float
            Wing zero-lift angle with Mach correction [rad]
    """
    
    def setup(self):
        # Inputs
        self.add_input('alpha_0', val=0.0, units='rad', 
                      desc='airfoil zero-lift angle')
        self.add_input('delta_alpha_0', val=0.0, units='rad', 
                      desc='Change in zero-lift angle with respect to wing twist')
        self.add_input('epsilon_t', val=0.0, units='rad', 
                      desc='twist angle')
        self.add_input('alpha_0_M', val=0.0, units='rad', 
                      desc='Zero-lift angle at current Mach')
        self.add_input('alpha_0_M03', val=0.0, units='rad', 
                      desc='Reference zero-lift angle at M=0.3')
        
        # Outputs
        self.add_output('alpha_0_W', val=0.0, units='rad', 
                       desc='Wing zero-lift angle with Mach correction')
        
        # Declare partials
        self.declare_partials('alpha_0_W', ['alpha_0', 'delta_alpha_0', 
                                          'epsilon_t', 'alpha_0_M', 'alpha_0_M03'])
        
    def compute(self, inputs, outputs):
        alpha_0 = inputs['alpha_0']
        delta_alpha_0 = inputs['delta_alpha_0']
        epsilon_t = inputs['epsilon_t']
        alpha_0_M = inputs['alpha_0_M']
        alpha_0_M03 = inputs['alpha_0_M03']
        
        # Equation from image
        outputs['alpha_0_W'] = (alpha_0 + 
                               (delta_alpha_0/epsilon_t) * epsilon_t * 
                               (alpha_0_M/alpha_0_M03))
        
    def compute_partials(self, inputs, partials):
        epsilon_t = inputs['epsilon_t']
        alpha_0_M = inputs['alpha_0_M']
        alpha_0_M03 = inputs['alpha_0_M03']
        
        # Derivatives
        partials['alpha_0_W', 'alpha_0'] = 1.0
        partials['alpha_0_W', 'delta_alpha_0'] = (alpha_0_M/alpha_0_M03)
        partials['alpha_0_W', 'epsilon_t'] = (inputs['delta_alpha_0'] * 
                                             alpha_0_M/alpha_0_M03)
        partials['alpha_0_W', 'alpha_0_M'] = ((inputs['delta_alpha_0']/epsilon_t) * 
                                             epsilon_t/alpha_0_M03)
        partials['alpha_0_W', 'alpha_0_M03'] = -((inputs['delta_alpha_0']/epsilon_t) * 
                                                epsilon_t * alpha_0_M/alpha_0_M03**2) 