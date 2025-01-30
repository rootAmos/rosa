import openmdao.api as om
import numpy as np

class FormFactor(om.ExplicitComponent):
    """
    Calculates the form factor (FF) for a wing using equation 13.22.
    
    Inputs:
        t_c : float
            Thickness to chord ratio [-]
        x_t : float
            Position of maximum thickness [-]
        M : float
            Mach number [-]
        phi_m : float
            Wing sweep angle at maximum thickness [rad]
    
    Outputs:
        FF : float
            Form factor [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('t_c', val=0.0, desc='Thickness to chord ratio')
        self.add_input('x_t', val=0.3, desc='Position of maximum thickness')
        self.add_input('M', val=0.0, desc='Mach number')
        self.add_input('phi_m', val=0.0, units='rad', desc='Sweep angle at max thickness')
        
        # Outputs
        self.add_output('FF', val=1.0, desc='Form factor')
        
        # Declare partials
        self.declare_partials('FF', ['t_c', 'x_t', 'M', 'phi_m'])
        
    def compute(self, inputs, outputs):
        t_c = inputs['t_c']
        x_t = inputs['x_t']
        M = inputs['M']
        phi_m = inputs['phi_m']
        
        # Form factor equation 13.22
        term1 = 1 + (0.6/x_t)*(t_c) + 100*(t_c)**4
        term2 = 1.34 * M**0.18 * (np.cos(phi_m))**0.28
        
        outputs['FF'] = term1 * term2
        
    def compute_partials(self, inputs, partials):
        t_c = inputs['t_c']
        x_t = inputs['x_t']
        M = inputs['M']
        phi_m = inputs['phi_m']
        
        # Intermediate terms
        term1 = 1 + (0.6/x_t)*(t_c) + 100*(t_c)**4
        term2 = 1.34 * M**0.18 * (np.cos(phi_m))**0.28
        
        # Partial derivatives
        dterm1_dtc = 0.6/x_t + 400*(t_c)**3
        dterm1_dxt = -0.6*(t_c)/(x_t**2)
        
        dterm2_dM = 1.34 * 0.18 * M**(-0.82) * (np.cos(phi_m))**0.28
        dterm2_dphi = 1.34 * M**0.18 * 0.28 * (np.cos(phi_m))**(-0.72) * (-np.sin(phi_m))
        
        partials['FF', 't_c'] = dterm1_dtc * term2
        partials['FF', 'x_t'] = dterm1_dxt * term2
        partials['FF', 'M'] = term1 * dterm2_dM
        partials['FF', 'phi_m'] = term1 * dterm2_dphi 