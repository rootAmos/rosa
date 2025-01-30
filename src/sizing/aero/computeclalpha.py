import openmdao.api as om
import numpy as np

class LiftCurveSlope(om.ExplicitComponent):
    """
    Calculates the lift curve slope for a lifting surface using compressibility corrections.
    
    Inputs:
        AR : float
            Aspect ratio [-]
        M : float
            Mach number [-]
        phi_50 : float
            50% chord sweep angle [rad]
        cl_alpha : float
            Airfoil section lift curve slope [1/rad]
    
    Outputs:
        CL_alpha : float
            Wing lift curve slope [1/rad]
    """
    
    def setup(self):
        # Inputs
        self.add_input('AR', val=0.0, desc='Aspect ratio')
        self.add_input('M', val=0.0, desc='Mach number')
        self.add_input('phi_50', val=0.0, units='rad', desc='50% chord sweep angle')
        self.add_input('cl_alpha', val=2*np.pi, units='1/rad', desc='Airfoil lift curve slope')
        
        # Outputs
        self.add_output('CL_alpha', val=0.0, units='1/rad', desc='Wing lift curve slope')
        
        # Declare partials
        self.declare_partials('CL_alpha', ['AR', 'M', 'phi_50', 'cl_alpha'])
        
    def compute(self, inputs, outputs):
        AR = inputs['AR']
        M = inputs['M']
        phi_50 = inputs['phi_50']
        cl_alpha = inputs['cl_alpha']
        
        # Compute beta (Prandtl-Glauert correction)
        beta = np.sqrt(1 - M**2)
        
        # Compute kappa
        kappa = cl_alpha / (2*np.pi/beta)
        
        # Compute CL_alpha
        term1 = 2*np.pi*AR
        term2 = 2 + np.sqrt((AR**2 * beta**2 / kappa**2) * (1 + np.tan(phi_50)**2/beta**2)) + 4
        
        outputs['CL_alpha'] = term1/term2
        
    def compute_partials(self, inputs, partials):
        AR = inputs['AR']
        M = inputs['M']
        phi_50 = inputs['phi_50']
        cl_alpha = inputs['cl_alpha']
        
        beta = np.sqrt(1 - M**2)
        kappa = cl_alpha / (2*np.pi/beta)
        
        # Base terms
        term1 = 2*np.pi*AR
        sqrt_term = np.sqrt((AR**2 * beta**2 / kappa**2) * (1 + np.tan(phi_50)**2/beta**2))
        term2 = 2 + sqrt_term + 4
        
        # Partial derivatives
        d_beta_dM = -M/beta
        
        # With respect to AR
        d_sqrt_dAR = (AR*beta**2/kappa**2 * (1 + np.tan(phi_50)**2/beta**2)) / sqrt_term
        partials['CL_alpha', 'AR'] = 2*np.pi/term2 - term1/(term2**2) * d_sqrt_dAR
        
        # With respect to M
        d_sqrt_dM = (2*AR**2*beta*d_beta_dM/kappa**2 * (1 + np.tan(phi_50)**2/beta**2) - 
                    2*AR**2*beta**2*np.tan(phi_50)**2/(kappa**2*beta**3)*d_beta_dM) / sqrt_term
        partials['CL_alpha', 'M'] = -term1/(term2**2) * d_sqrt_dM
        
        # With respect to phi_50
        d_sqrt_dphi = (AR**2*beta**2/kappa**2 * 2*np.tan(phi_50)/(beta**2*np.cos(phi_50)**2)) / sqrt_term
        partials['CL_alpha', 'phi_50'] = -term1/(term2**2) * d_sqrt_dphi
        
        # With respect to cl_alpha
        d_sqrt_dcl = (-2*AR**2*beta**2/(kappa**3) * (1 + np.tan(phi_50)**2/beta**2)) / sqrt_term * (-kappa**2/(cl_alpha))
        partials['CL_alpha', 'cl_alpha'] = -term1/(term2**2) * d_sqrt_dcl
