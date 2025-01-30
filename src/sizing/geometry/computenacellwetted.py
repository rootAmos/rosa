import openmdao.api as om
import numpy as np

class NacelleWettedArea(om.ExplicitComponent):
    """
    Calculates the wetted area of a ducted fan nacelle using equation 13.12.
    
    Inputs:
        l_n : float
            Nacelle length [m]
        D_n : float
            Nacelle diameter [m]
        l_1 : float
            Inlet length [m]
        D_hl : float
            Highlight diameter [m]
        D_ef : float
            Exit diameter [m]
    
    Outputs:
        S_wet_fan_cowl : float
            Nacelle wetted area [m^2]
    """
    
    def setup(self):
        # Inputs
        self.add_input('l_n', val=0.0, units='m', desc='Nacelle length')
        self.add_input('D_n', val=0.0, units='m', desc='Nacelle diameter')
        self.add_input('l_1', val=0.0, units='m', desc='Inlet length')
        self.add_input('D_hl', val=0.0, units='m', desc='Highlight diameter')
        self.add_input('D_ef', val=0.0, units='m', desc='Exit diameter')
        
        # Output
        self.add_output('S_wet_fan_cowl', val=0.0, units='m**2', 
                       desc='Nacelle wetted area')
        
        # Declare partials
        self.declare_partials('S_wet_fan_cowl', 
                            ['l_n', 'D_n', 'l_1', 'D_hl', 'D_ef'])
        
    def compute(self, inputs, outputs):
        l_n = inputs['l_n']
        D_n = inputs['D_n']
        l_1 = inputs['l_1']
        D_hl = inputs['D_hl']
        D_ef = inputs['D_ef']
        
        # Compute terms in brackets
        term1 = 2.0
        term2 = 0.35 * l_1/l_n
        term3 = 0.8 * l_1 * D_hl/(l_n * D_n)
        term4 = 1.15 * (1 - l_1/l_n) * D_ef/D_n
        
        # Compute wetted area using equation 13.12
        outputs['S_wet_fan_cowl'] = l_n * D_n * (term1 + term2 + term3 + term4)
        
    def compute_partials(self, inputs, partials):
        l_n = inputs['l_n']
        D_n = inputs['D_n']
        l_1 = inputs['l_1']
        D_hl = inputs['D_hl']
        D_ef = inputs['D_ef']
        
        # Partial derivatives
        # With respect to l_n
        partials['S_wet_fan_cowl', 'l_n'] = (
            D_n * (2.0 - 0.35*l_1/l_n**2 - 
                  0.8*l_1*D_hl/(l_n**2*D_n) + 
                  1.15*l_1*D_ef/(l_n**2*D_n))
        )
        
        # With respect to D_n
        partials['S_wet_fan_cowl', 'D_n'] = (
            l_n * (2.0 + 0.35*l_1/l_n - 
                  0.8*l_1*D_hl/(l_n*D_n**2) - 
                  1.15*(1-l_1/l_n)*D_ef/D_n**2)
        )
        
        # With respect to l_1
        partials['S_wet_fan_cowl', 'l_1'] = (
            l_n * D_n * (0.35/l_n + 
                        0.8*D_hl/(l_n*D_n) - 
                        1.15*D_ef/(l_n*D_n))
        )
        
        # With respect to D_hl
        partials['S_wet_fan_cowl', 'D_hl'] = (
            l_n * D_n * 0.8 * l_1/(l_n*D_n)
        )
        
        # With respect to D_ef
        partials['S_wet_fan_cowl', 'D_ef'] = (
            l_n * D_n * 1.15 * (1-l_1/l_n)/D_n
        ) 