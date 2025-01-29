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
        self.add_input('deps_c_da', val=0.0,
                      desc='Canard downwash derivative')
        self.add_input('S_c', val=0.0, units='m**2',
                      desc='Canard reference area')
        self.add_input('S_w', val=0.0, units='m**2',
                      desc='Wing reference area')
        
        # Outputs
        self.add_output('CL_alpha_c_eff', val=0.0, units='1/rad',
                       desc='Effective canard lift curve slope')
        
        # Partials
        self.declare_partials('CL_alpha_c_eff', ['CL_alpha_c', 'deps_c_da', 'S_c', 'S_w'])
        
    def compute(self, inputs, outputs):
        CL_alpha_c = inputs['CL_alpha_c']
        deps_c_da = inputs['deps_c_da']
        S_c = inputs['S_c']
        S_w = inputs['S_w']
        
        outputs['CL_alpha_c_eff'] = CL_alpha_c * (S_c/S_w) * (1 + deps_c_da)
        
    def compute_partials(self, inputs, partials):
        S_c = inputs['S_c']
        S_w = inputs['S_w']
        deps_c_da = inputs['deps_c_da']
        CL_alpha_c = inputs['CL_alpha_c']
        
        partials['CL_alpha_c_eff', 'CL_alpha_c'] = (S_c/S_w) * (1 + deps_c_da)
        partials['CL_alpha_c_eff', 'deps_c_da'] = CL_alpha_c * (S_c/S_w)
        partials['CL_alpha_c_eff', 'S_c'] = CL_alpha_c * (1 + deps_c_da) / S_w
        partials['CL_alpha_c_eff', 'S_w'] = -CL_alpha_c * S_c * (1 + deps_c_da) / (S_w**2) 