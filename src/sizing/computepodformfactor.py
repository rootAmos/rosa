import openmdao.api as om
import numpy as np

class PodFormFactor(om.ExplicitComponent):
    """
    Calculates the form factor for a pod/fuselage using equation 13.23.
    
    Inputs:
        l_F : float
            Pod length [m]
        d_F : float
            Pod diameter [m]
    
    Outputs:
        FF_F : float
            Pod form factor [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('l_F', val=0.0, units='m', desc='Pod length')
        self.add_input('d_F', val=0.0, units='m', desc='Pod diameter')
        
        # Output
        self.add_output('FF_F', val=0.0, desc='Pod form factor')
        
        # Declare partials
        self.declare_partials('FF_F', ['l_F', 'd_F'])
        
    def compute(self, inputs, outputs):
        l_F = inputs['l_F']
        d_F = inputs['d_F']
        
        # Compute fineness ratio
        fineness_ratio = l_F / d_F
        
        # Compute form factor using equation 13.23
        outputs['FF_F'] = (1.0 + 
                          60.0/(fineness_ratio**3) + 
                          fineness_ratio/400.0)
        
    def compute_partials(self, inputs, partials):
        l_F = inputs['l_F']
        d_F = inputs['d_F']
        
        # Partial derivatives
        # With respect to l_F
        partials['FF_F', 'l_F'] = (
            -180.0/(l_F**3/d_F**3) +
            1.0/(400.0*d_F)
        )
        
        # With respect to d_F
        partials['FF_F', 'd_F'] = (
            180.0*l_F/(d_F**4) -
            l_F/(400.0*d_F**2)
        ) 