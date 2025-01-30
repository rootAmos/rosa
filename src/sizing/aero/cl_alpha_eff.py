import openmdao.api as om
import numpy as np

class ComputeClAlphaCanard(om.ExplicitComponent):
    """
    Calculates the effective lift curve slope for the canard, accounting for area ratio and downwash.
    
    Inputs:
        CL_alpha_c : float
            Canard lift curve slope [1/rad]
        deps_c_da : float
            Change in canard downwash angle with respect to angle of attack [-]
        S_c : float
            Canard reference area [m^2]
        S_w : float
            Wing reference area [m^2]
    
    Outputs:
        CL_alpha_c_eff : float
            Effective canard lift curve slope [1/rad]
    """
    
    def setup(self):
        # Inputs
        self.add_input('CL_alpha_c', val=0.0, units='1/rad',
                      desc='Canard lift curve slope')
        self.add_input('d_eps_c_d_alpha', val=0.0,
                      desc='Canard downwash derivative')

        
        # Outputs
        self.add_output('CL_alpha_c_eff', val=0.0, units='1/rad',
                       desc='Effective canard lift curve slope')
        
        # Partials
        self.declare_partials('CL_alpha_c_eff', ['CL_alpha_c', 'd_eps_c_d_alpha'])
        
    def compute(self, inputs, outputs):
        CL_alpha_c = inputs['CL_alpha_c']
        d_eps_c_d_alpha = inputs['d_eps_c_d_alpha']

        
        outputs['CL_alpha_c_eff'] = CL_alpha_c  * (1 + d_eps_c_d_alpha)
        
    def compute_partials(self, inputs, partials):
        d_eps_c_d_alpha = inputs['d_eps_c_d_alpha']
        CL_alpha_c = inputs['CL_alpha_c']
        
        partials['CL_alpha_c_eff', 'CL_alpha_c'] = (1 + d_eps_c_d_alpha)
        partials['CL_alpha_c_eff', 'd_eps_c_d_alpha'] = CL_alpha_c 

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
        self.add_input('d_eps_w_d_alpha', val=0.0, 
                      desc='Wing downwash derivative')
        
        # Outputs
        self.add_output('CL_alpha_w_eff', val=0.0, units='1/rad',
                       desc='Effective wing lift curve slope')
        
        # Partials
        self.declare_partials('CL_alpha_w_eff', ['CL_alpha_w', 'd_eps_w_d_alpha'])
        
    def compute(self, inputs, outputs):
        CL_alpha_w = inputs['CL_alpha_w']
        d_eps_w_d_alpha = inputs['d_eps_w_d_alpha']
        
        outputs['CL_alpha_w_eff'] = CL_alpha_w * (1 - d_eps_w_d_alpha)
        
    def compute_partials(self, inputs, partials):
        partials['CL_alpha_w_eff', 'CL_alpha_w'] = 1 - inputs['d_eps_w_d_alpha']
        partials['CL_alpha_w_eff', 'd_eps_w_d_alpha'] = -inputs['CL_alpha_w'] 