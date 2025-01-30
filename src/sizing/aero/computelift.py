import openmdao.api as om
import numpy as np

class LiftCoefficients(om.ExplicitComponent):
    """
    Calculates lift coefficients for wing and canard.
    
    Inputs:
        CL_alpha_w : float
            Wing lift curve slope [1/rad]
        alpha : float
            Angle of attack [rad]
        i_w : float
            Wing incidence angle [rad]
        alpha_0_w : float
            Wing zero-lift angle of attack [rad]
        epsilon : float
            Downwash angle [rad]
        CL_alpha_c : float
            Canard lift curve slope [1/rad]
        i_c : float
            Canard incidence angle [rad]
        alpha_0_c : float
            Canard zero-lift angle of attack [rad]
    
    Outputs:
        CL_w : float
            Wing lift coefficient [-]
        CL_c : float
            Canard lift coefficient [-]
    """
    
    def setup(self):
        # Wing inputs
        self.add_input('CL_alpha_w', val=0.0, units='1/rad', desc='Wing lift curve slope')
        self.add_input('alpha', val=0.0, units='rad', desc='Angle of attack')
        self.add_input('i_w', val=0.0, units='rad', desc='Wing incidence angle')
        self.add_input('alpha_0_w', val=0.0, units='rad', desc='Wing zero-lift angle')
        self.add_input('epsilon', val=0.0, units='rad', desc='Downwash angle')
        
        # Canard inputs
        self.add_input('CL_alpha_c', val=0.0, units='1/rad', desc='Canard lift curve slope')
        self.add_input('i_c', val=0.0, units='rad', desc='Canard incidence angle')
        self.add_input('alpha_0_c', val=0.0, units='rad', desc='Canard zero-lift angle')
        
        # Outputs
        self.add_output('CL_w', val=0.0, desc='Wing lift coefficient')
        self.add_output('CL_c', val=0.0, desc='Canard lift coefficient')
        
        # Declare partials
        self.declare_partials('CL_w', ['CL_alpha_w', 'alpha', 'i_w', 'alpha_0_w', 'epsilon'])
        self.declare_partials('CL_c', ['CL_alpha_c', 'alpha', 'i_c', 'alpha_0_c'])
        
    def compute(self, inputs, outputs):
        # Wing lift (with downwash)
        outputs['CL_w'] = inputs['CL_alpha_w'] * (
            inputs['alpha'] + inputs['i_w'] - inputs['epsilon'] - inputs['alpha_0_w']
        )
        
        # Canard lift (no downwash)
        outputs['CL_c'] = inputs['CL_alpha_c'] * (
            inputs['alpha'] + inputs['i_c'] - inputs['alpha_0_c']
        )
        
    def compute_partials(self, inputs, partials):
        # Wing derivatives
        partials['CL_w', 'CL_alpha_w'] = (
            inputs['alpha'] + inputs['i_w'] - inputs['epsilon'] - inputs['alpha_0_w']
        )
        partials['CL_w', 'alpha'] = inputs['CL_alpha_w']
        partials['CL_w', 'i_w'] = inputs['CL_alpha_w']
        partials['CL_w', 'alpha_0_w'] = -inputs['CL_alpha_w']
        partials['CL_w', 'epsilon'] = -inputs['CL_alpha_w']
        
        # Canard derivatives
        partials['CL_c', 'CL_alpha_c'] = (
            inputs['alpha'] + inputs['i_c'] - inputs['alpha_0_c']
        )
        partials['CL_c', 'alpha'] = inputs['CL_alpha_c']
        partials['CL_c', 'i_c'] = inputs['CL_alpha_c']
        partials['CL_c', 'alpha_0_c'] = -inputs['CL_alpha_c'] 