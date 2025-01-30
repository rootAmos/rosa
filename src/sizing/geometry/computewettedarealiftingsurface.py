import openmdao.api as om
import numpy as np

class ComputeWingWettedAreaLiftingSurface(om.ExplicitComponent):
    """
    Calculates the wetted area of a wing.
    
    Inputs:
        S_exp : float
            Exposed planform area [m^2]
        t_c : float
            Thickness to chord ratio [-]
        tau : float
            Airfoil thickness location parameter [-]
        lambda_w : float
            Wing taper ratio [-]
    
    Outputs:
        S_wet : float
            Wetted area [m^2]
    """
    
    def setup(self):
        # Inputs
        self.add_input('S_exp', val=0.0, units='m**2', 
                      desc='Exposed planform area')
        self.add_input('t_c', val=0.0, 
                      desc='Thickness to chord ratio')
        self.add_input('tau', val=0.0, 
                      desc='Airfoil thickness location parameter')
        self.add_input('lambda_w', val=0.0, 
                      desc='Wing taper ratio')
        
        # Outputs
        self.add_output('S_wet', val=0.0, units='m**2', 
                       desc='Wetted area')
        
        # Declare partials
        self.declare_partials('S_wet', ['S_exp', 't_c', 'tau', 'lambda_w'])
        
    def compute(self, inputs, outputs):
        S_exp = inputs['S_exp']
        t_c = inputs['t_c']
        tau = inputs['tau']
        lambda_w = inputs['lambda_w']
        
        # Equation from image
        outputs['S_wet'] = 2 * S_exp * (1 + 0.25*(t_c) * ((1 + tau*lambda_w)/(1 + lambda_w)))
        
    def compute_partials(self, inputs, partials):
        S_exp = inputs['S_exp']
        t_c = inputs['t_c']
        tau = inputs['tau']
        lambda_w = inputs['lambda_w']
        
        # Partial derivatives
        partials['S_wet', 'S_exp'] = 2 * (1 + 0.25*(t_c) * ((1 + tau*lambda_w)/(1 + lambda_w)))
        
        partials['S_wet', 't_c'] = 2 * S_exp * 0.25 * ((1 + tau*lambda_w)/(1 + lambda_w))
        
        partials['S_wet', 'tau'] = 2 * S_exp * 0.25 * t_c * (lambda_w/(1 + lambda_w))
        
        partials['S_wet', 'lambda_w'] = (2 * S_exp * 0.25 * t_c * 
                                        (tau*(1 + lambda_w) - (1 + tau*lambda_w))/
                                        (1 + lambda_w)**2) 