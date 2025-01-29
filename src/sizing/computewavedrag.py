import openmdao.api as om
import numpy as np

class WaveDrag(om.ExplicitComponent):
    """
    Calculates the wave drag increment using equation 13.25.
    
    Inputs:
        M : float
            Mach number [-]
        M_crit : float
            Critical Mach number [-]
        a : float
            Wave drag coefficient a [-]
        b : float
            Wave drag coefficient b [-]
    
    Outputs:
        CD_wave : float
            Wave drag coefficient increment [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('M', val=0.0, desc='Mach number')
        self.add_input('M_crit', val=0.7, desc='Critical Mach number')
        self.add_input('a', val=0.0, desc='Wave drag coefficient a')
        self.add_input('b', val=3.0, desc='Wave drag coefficient b')
        
        # Outputs
        self.add_output('CD_wave', val=0.0, desc='Wave drag coefficient increment')
        
        # Declare partials
        self.declare_partials('CD_wave', ['M', 'M_crit', 'a', 'b'])
        
    def compute(self, inputs, outputs):
        M = inputs['M']
        M_crit = inputs['M_crit']
        a = inputs['a']
        b = inputs['b']
        
        # Only compute wave drag if M > M_crit
        if M > M_crit:
            outputs['CD_wave'] = a * ((M/M_crit) - 1)**b
        else:
            outputs['CD_wave'] = 0.0
        
    def compute_partials(self, inputs, partials):
        M = inputs['M']
        M_crit = inputs['M_crit']
        a = inputs['a']
        b = inputs['b']
        
        if M > M_crit:
            # Partial with respect to M
            partials['CD_wave', 'M'] = (a * b * ((M/M_crit) - 1)**(b-1)) / M_crit
            
            # Partial with respect to M_crit
            partials['CD_wave', 'M_crit'] = -a * b * M * ((M/M_crit) - 1)**(b-1) / (M_crit**2)
            
            # Partial with respect to a
            partials['CD_wave', 'a'] = ((M/M_crit) - 1)**b
            
            # Partial with respect to b
            term = ((M/M_crit) - 1)
            if term > 0:  # Avoid log(negative)
                partials['CD_wave', 'b'] = a * (term**b) * np.log(term)
            else:
                partials['CD_wave', 'b'] = 0.0
        else:
            # Zero derivatives when M <= M_crit
            partials['CD_wave', 'M'] = 0.0
            partials['CD_wave', 'M_crit'] = 0.0
            partials['CD_wave', 'a'] = 0.0
            partials['CD_wave', 'b'] = 0.0 