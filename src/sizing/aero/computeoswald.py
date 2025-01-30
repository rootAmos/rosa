import openmdao.api as om
import numpy as np

class OswaldEfficiency(om.ExplicitComponent):
    """
    Calculates the Oswald efficiency factor for a lifting surface using equation 13.26.
    
    Inputs:
        M : float
            Flight Mach number [-]
        A : float
            Effective aspect ratio [-]
        t_c : float
            Relative airfoil thickness [-]
        phi_25 : float
            Quarter-chord sweep angle [rad]
        N_e : float
            Number of engines on the surface [-]
        taper : float
            Wing taper ratio [-]
    
    Outputs:
        e : float
            Oswald efficiency factor [-]
        f_lambda : float
            Taper correction factor [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('M', val=0.0, desc='Flight Mach number')
        self.add_input('A', val=0.0, desc='Effective aspect ratio')
        self.add_input('t_c', val=0.0, desc='Relative airfoil thickness')
        self.add_input('phi_25', val=0.0, units='rad', desc='Quarter-chord sweep angle')
        self.add_input('N_e', val=0.0, desc='Number of engines on surface')
        self.add_input('taper', val=1.0, desc='Taper ratio')
        
        # Outputs
        self.add_output('e', val=0.0, desc='Oswald efficiency factor')
        self.add_output('f_lambda', val=0.0062, desc='Taper correction factor')
        
        # Declare partials
        self.declare_partials('e', ['M', 'A', 't_c', 'phi_25', 'N_e', 'taper'])
        self.declare_partials('f_lambda', ['taper'])
        
    def compute(self, inputs, outputs):
        M = inputs['M']
        A = inputs['A']
        t_c = inputs['t_c']
        phi_25 = inputs['phi_25']
        N_e = inputs['N_e']
        taper = inputs['taper']
        
        # Compute f(λ) using equation 13.27
        outputs['f_lambda'] = 0.005 * (1 + 1.5 * (taper - 0.6)**2)
        
        # Compute terms in denominator
        term1 = 1 + 0.12 * M**6
        
        term2 = 1 + (
            (0.142 + outputs['f_lambda'] * A * (10 * t_c)**0.33) / 
            (np.cos(phi_25))**2 +
            0.1 * (3 * N_e + 1) / (4 + A)**0.8
        )
        
        # Compute Oswald factor using equation 13.26
        outputs['e'] = 1.0 / (term1 * term2)
        
    def compute_partials(self, inputs, partials):
        M = inputs['M']
        A = inputs['A']
        t_c = inputs['t_c']
        phi_25 = inputs['phi_25']
        N_e = inputs['N_e']
        taper = inputs['taper']
        
        # Compute f(λ) and its derivative
        f_lambda = 0.005 * (1 + 1.5 * (taper - 0.6)**2)
        df_lambda_dtaper = 0.005 * 3 * (taper - 0.6)
        
        # Common terms
        term1 = 1 + 0.12 * M**6
        cos_phi_sq = np.cos(phi_25)**2
        term2_1 = (0.142 + f_lambda * A * (10 * t_c)**0.33) / cos_phi_sq
        term2_2 = 0.1 * (3 * N_e + 1) / (4 + A)**0.8
        term2 = 1 + term2_1 + term2_2
        e = 1.0 / (term1 * term2)
        
        # Partial derivatives
        partials['f_lambda', 'taper'] = df_lambda_dtaper
        
        partials['e', 'M'] = -e * (0.72 * M**5) * term2 / term1
        
        partials['e', 'A'] = -e * (
            f_lambda * (10 * t_c)**0.33 / cos_phi_sq -
            0.08 * (3 * N_e + 1) / (4 + A)**1.8
        )
        
        partials['e', 't_c'] = -e * (
            0.33 * f_lambda * A * (10 * t_c)**(-0.67) * 10 / cos_phi_sq
        )
        
        partials['e', 'phi_25'] = -e * (
            2 * (0.142 + f_lambda * A * (10 * t_c)**0.33) * 
            np.sin(phi_25) / (np.cos(phi_25)**3)
        )
        
        partials['e', 'N_e'] = -e * (
            0.3 / (4 + A)**0.8
        )
        
        partials['e', 'taper'] = -e * (
            df_lambda_dtaper * A * (10 * t_c)**0.33 / cos_phi_sq
        ) 