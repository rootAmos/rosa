import openmdao.api as om
import numpy as np

class NacelleFormFactor(om.ExplicitComponent):
    """
    Calculates the form factor for a nacelle using equation 13.24.
    
    Inputs:
        l_N : float
            Nacelle length [m]
        d_N : float
            Nacelle diameter [m]
    
    Outputs:
        FF_N : float
            Nacelle form factor [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('l_N', val=0.0, units='m', desc='Nacelle length')
        self.add_input('d_N', val=0.0, units='m', desc='Nacelle diameter')
        
        # Output
        self.add_output('FF_N', val=0.0, desc='Nacelle form factor')
        
        # Declare partials
        self.declare_partials('FF_N', ['l_N', 'd_N'])
        
    def compute(self, inputs, outputs):
        l_N = inputs['l_N']
        d_N = inputs['d_N']
        
        # Compute fineness ratio
        fineness_ratio = l_N / d_N
        
        # Compute form factor using equation 13.24
        outputs['FF_N'] = 1.0 + 0.35/fineness_ratio
        
    def compute_partials(self, inputs, partials):
        l_N = inputs['l_N']
        d_N = inputs['d_N']
        
        # Partial derivatives
        partials['FF_N', 'l_N'] = -0.35/(l_N**2/d_N)
        partials['FF_N', 'd_N'] = 0.35*l_N/(d_N**2) 