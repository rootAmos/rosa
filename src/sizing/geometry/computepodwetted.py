import openmdao.api as om
import numpy as np

class PodWettedArea(om.ExplicitComponent):
    """
    Calculates the wetted area of a pod/plug using equation 13.14.
    
    Inputs:
        l_p : float
            Pod length [m]
        D_p : float
            Pod diameter [m]
    
    Outputs:
        S_wet_plug : float
            Pod wetted area [m^2]
    """
    
    def setup(self):
        # Inputs
        self.add_input('l_p', val=0.0, units='m', desc='Pod length')
        self.add_input('D_p', val=0.0, units='m', desc='Pod diameter')
        
        # Output
        self.add_output('S_wet_plug', val=0.0, units='m**2', desc='Pod wetted area')
        
        # Declare partials
        self.declare_partials('S_wet_plug', ['l_p', 'D_p'])
        
    def compute(self, inputs, outputs):
        l_p = inputs['l_p']
        D_p = inputs['D_p']
        
        # Compute wetted area using equation 13.14
        outputs['S_wet_plug'] = 0.7 * np.pi * l_p * D_p
        
    def compute_partials(self, inputs, partials):
        l_p = inputs['l_p']
        D_p = inputs['D_p']
        
        # Derivatives
        partials['S_wet_plug', 'l_p'] = 0.7 * np.pi * D_p
        partials['S_wet_plug', 'D_p'] = 0.7 * np.pi * l_p 