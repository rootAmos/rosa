import openmdao.api as om
import numpy as np

class Downwash(om.ExplicitComponent):
    """
    Calculates the downwash angle experienced by the wing due to the canard.
    
    Inputs:
        CL_c : float
            Canard lift coefficient [-]
        A : float
            Canard aspect ratio [-]
        delta_1 : float
            First downwash correction factor [-]
        delta_2 : float
            Second downwash correction factor [-]
    
    Outputs:
        epsilon : float
            Downwash angle [rad]
    """
    
    def setup(self):
        # Inputs
        self.add_input('CL_c', val=0.0, desc='Canard lift coefficient')
        self.add_input('A', val=0.0, desc='Canard aspect ratio')
        self.add_input('delta_1', val=1.0, desc='First downwash correction factor')
        self.add_input('delta_2', val=1.0, desc='Second downwash correction factor')
        
        # Outputs
        self.add_output('epsilon', val=0.0, units='rad', desc='Downwash angle')
        
        # Declare partials
        self.declare_partials('epsilon', ['CL_c', 'A', 'delta_1', 'delta_2'])
        
    def compute(self, inputs, outputs):
        CL_c = inputs['CL_c']
        A = inputs['A']
        delta_1 = inputs['delta_1']
        delta_2 = inputs['delta_2']
        
        # Compute sqrt term once
        sqrt_term = np.sqrt(CL_c**2 + 1)
        
        # Compute the downwash angle
        outputs['epsilon'] = (CL_c/(np.pi * A)) * (
            (1 - CL_c/sqrt_term) * delta_1 + 
            (CL_c/sqrt_term) * delta_2
        )
        
    def compute_partials(self, inputs, partials):
        CL_c = inputs['CL_c']
        A = inputs['A']
        delta_1 = inputs['delta_1']
        delta_2 = inputs['delta_2']
        
        sqrt_term = np.sqrt(CL_c**2 + 1)
        
        # Derivative with respect to CL_c
        d_sqrt_term_dCL = CL_c/sqrt_term
        
        depsilon_dCL = (1/(np.pi * A)) * (
            (1 - CL_c/sqrt_term) * delta_1 + 
            (CL_c/sqrt_term) * delta_2 +
            CL_c * (
                (-1/sqrt_term + CL_c**2/(sqrt_term**3)) * delta_1 +
                (1/sqrt_term - CL_c**2/(sqrt_term**3)) * delta_2
            )
        )
        
        # Derivative with respect to A
        depsilon_dA = -CL_c/(np.pi * A**2) * (
            (1 - CL_c/sqrt_term) * delta_1 + 
            (CL_c/sqrt_term) * delta_2
        )
        
        # Derivative with respect to delta_1
        depsilon_ddelta1 = (CL_c/(np.pi * A)) * (1 - CL_c/sqrt_term)
        
        # Derivative with respect to delta_2
        depsilon_ddelta2 = (CL_c/(np.pi * A)) * (CL_c/sqrt_term)
        
        # Assign derivatives
        partials['epsilon', 'CL_c'] = depsilon_dCL
        partials['epsilon', 'A'] = depsilon_dA
        partials['epsilon', 'delta_1'] = depsilon_ddelta1
        partials['epsilon', 'delta_2'] = depsilon_ddelta2 