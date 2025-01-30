import openmdao.api as om
import numpy as np

class ZeroLiftAngle(om.ExplicitComponent):
    """
    Calculates the zero-lift angle of attack with Mach number corrections.
    
    Inputs:
        alpha0_airfoil : float
            Basic zero-lift angle of attack at M=0.3 [rad]
        delta_alpha_0_d_twist : float
            Change in zero-lift angle due to wing twist [rad]
        epsilon_t : float
            Twist angle [rad]
    
    Outputs:
        alpha0 : float
            lifting surface zero-lift angle with Mach correction [rad]
    """
    
    def setup(self):
        # Inputs
        self.add_input('alpha0_airfoil', val=0.0, units='rad', 
                      desc='airfoil zero-lift angle')
        self.add_input('d_alpha0_d_twist', val=0.0, units='rad', 
                      desc='Change in zero-lift angle with respect to wing twist')
        self.add_input('d_twist', val=0.0, units='rad', 
                      desc='twist angle')
        
        # Outputs
        self.add_output('alpha0', val=0.0, units='rad', 
                       desc='lifting surface zero-lift angle with Mach correction')
        
        # Declare partials
        self.declare_partials('alpha0', ['alpha0_airfoil', 'd_alpha0_d_twist', 
                                          'd_twist'])
        
    def compute(self, inputs, outputs):


        alpha0_airfoil = inputs['alpha0_airfoil']
        d_alpha0_d_twist = inputs['d_alpha0_d_twist']
        d_twist = inputs['d_twist']

        
        # Equation from image
        outputs['alpha0'] = (alpha0_airfoil + d_alpha0_d_twist * d_twist)
                               
        
    def compute_partials(self, inputs, partials):

        # Inputs
        d_alpha0_d_twist = inputs['d_alpha0_d_twist']
        d_twist = inputs['d_twist']
        alpha0_airfoil = inputs['alpha0_airfoil']
        
        # Derivatives
        partials['alpha0', 'alpha0_airfoil'] = 1.0
        partials['alpha0', 'd_alpha0_d_twist'] = d_twist
        partials['alpha0', 'd_twist'] = d_alpha0_d_twist
