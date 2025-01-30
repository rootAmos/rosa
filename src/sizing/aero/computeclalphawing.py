import openmdao.api as om
import numpy as np

class ComputeClAlphaWing(om.ExplicitComponent):
    """
    Calculates the effective lift curve slope for the wing, accounting for downwash.
    
    Inputs:
        CL_alpha_w : float
            Wing lift curve slope [1/rad]
        deps_w_da : float
            Change in wing downwash angle with respect to angle of attack [-]
    
    Outputs:
        CL_alpha_w_eff : float
            Effective wing lift curve slope [1/rad]
    """
    
    def setup(self):
        # Inputs
        self.add_input('CL_alpha_w', val=0.0, units='1/rad', 
                      desc='Wing lift curve slope')
        self.add_input('deps_w_da', val=0.0, 
                      desc='Wing downwash derivative')
        
        # Outputs
        self.add_output('CL_alpha_w_eff', val=0.0, units='1/rad',
                       desc='Effective wing lift curve slope')
        
        # Partials
        self.declare_partials('CL_alpha_w_eff', ['CL_alpha_w', 'deps_w_da'])
        
    def compute(self, inputs, outputs):
        CL_alpha_w = inputs['CL_alpha_w']
        deps_w_da = inputs['deps_w_da']
        
        outputs['CL_alpha_w_eff'] = CL_alpha_w * (1 - deps_w_da)
        
    def compute_partials(self, inputs, partials):
        partials['CL_alpha_w_eff', 'CL_alpha_w'] = 1 - inputs['deps_w_da']
        partials['CL_alpha_w_eff', 'deps_w_da'] = -inputs['CL_alpha_w'] 