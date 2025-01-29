import openmdao.api as om
import numpy as np

class SkinFriction(om.ExplicitComponent):
    """
    Calculates skin friction coefficient using a weighted combination 
    of laminar and turbulent contributions.
    
    Inputs:
        Re : float
            Reynolds number [-]
        M : float
            Mach number [-]
        k_lam : float
            Laminar flow fraction [-]
    
    Outputs:
        Cf : float
            Combined skin friction coefficient [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('Re', val=1e6, desc='Reynolds number')
        self.add_input('M', val=0.0, desc='Mach number')
        self.add_input('k_lam', val=0.0, desc='Laminar flow fraction')
        
        # Outputs
        self.add_output('Cf', val=0.0, desc='Skin friction coefficient')
        
        # Declare partials
        self.declare_partials('Cf', ['Re', 'M', 'k_lam'])
        
    def compute(self, inputs, outputs):
        Re = inputs['Re']
        M = inputs['M']
        k_lam = inputs['k_lam']
        
        # Calculate components
        Cf_lam = 1.328 / np.sqrt(Re)
        Cf_turb = 0.455 / (np.log10(Re)**2.58 * (1 + 0.144*M**2)**0.65)
        
        # Combine using weighted sum
        outputs['Cf'] = k_lam * Cf_lam + (1 - k_lam) * Cf_turb
        
    def compute_partials(self, inputs, partials):
        Re = inputs['Re']
        M = inputs['M']
        k_lam = inputs['k_lam']
        
        # Component derivatives
        Cf_lam = 1.328 / np.sqrt(Re)
        dCf_lam_dRe = -1.328 / (2 * Re * np.sqrt(Re))
        dCf_lam_dM = 0.0
        
        log_Re = np.log10(Re)
        Cf_turb = 0.455 / (log_Re**2.58 * (1 + 0.144*M**2)**0.65)
        dCf_turb_dRe = -0.455 * 2.58 / (log_Re**3.58 * (1 + 0.144*M**2)**0.65 * Re * np.log(10))
        dCf_turb_dM = -0.455 * 0.65 * 0.288 * M / (log_Re**2.58 * (1 + 0.144*M**2)**1.65)
        
        # Combined derivatives
        partials['Cf', 'Re'] = k_lam * dCf_lam_dRe + (1 - k_lam) * dCf_turb_dRe
        partials['Cf', 'M'] = k_lam * dCf_lam_dM + (1 - k_lam) * dCf_turb_dM
        partials['Cf', 'k_lam'] = Cf_lam - Cf_turb 