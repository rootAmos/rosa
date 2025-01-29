import openmdao.api as om
import numpy as np

class ZeroLiftDrag(om.ExplicitComponent):
    """
    Calculates the total zero-lift drag coefficient (CD0) for wing and canard using equation 13.15.
    
    Inputs:
        Cf_w : float
            Wing skin friction coefficient [-]
        FF_w : float
            Wing form factor [-]
        Q_w : float
            Wing interference factor [-]
        S_wet_w : float
            Wing wetted area [m^2]
        Cf_c : float
            Canard skin friction coefficient [-]
        FF_c : float
            Canard form factor [-]
        Q_c : float
            Canard interference factor [-]
        S_wet_c : float
            Canard wetted area [m^2]
        S_ref : float
            Reference area [m^2]
        CD_misc : float
            Miscellaneous drag coefficient [-]
        CD_LP : float
            Leakage and protuberance drag coefficient [-]
    
    Outputs:
        CD0 : float
            Total zero-lift drag coefficient [-]
    """
    
    def setup(self):
        # Wing inputs
        self.add_input('Cf_w', val=0.0, desc='Wing skin friction coefficient')
        self.add_input('FF_w', val=1.0, desc='Wing form factor')
        self.add_input('Q_w', val=1.0, desc='Wing interference factor')
        self.add_input('S_wet_w', val=0.0, units='m**2', desc='Wing wetted area')
        
        # Canard inputs
        self.add_input('Cf_c', val=0.0, desc='Canard skin friction coefficient')
        self.add_input('FF_c', val=1.0, desc='Canard form factor')
        self.add_input('Q_c', val=1.0, desc='Canard interference factor')
        self.add_input('S_wet_c', val=0.0, units='m**2', desc='Canard wetted area')
        
        # Common inputs
        self.add_input('S_ref', val=1.0, units='m**2', desc='Reference area')
        self.add_input('CD_misc', val=0.0, desc='Miscellaneous drag coefficient')
        self.add_input('CD_LP', val=0.0, desc='Leakage and protuberance drag')
        
        # Output
        self.add_output('CD0', val=0.0, desc='Zero-lift drag coefficient')
        
        # Declare partials
        self.declare_partials('CD0', ['Cf_w', 'FF_w', 'Q_w', 'S_wet_w',
                                    'Cf_c', 'FF_c', 'Q_c', 'S_wet_c',
                                    'S_ref', 'CD_misc', 'CD_LP'])
        
    def compute(self, inputs, outputs):
        # Wing component
        CD0_w = inputs['Cf_w'] * inputs['FF_w'] * inputs['Q_w'] * inputs['S_wet_w']/inputs['S_ref']
        
        # Canard component
        CD0_c = inputs['Cf_c'] * inputs['FF_c'] * inputs['Q_c'] * inputs['S_wet_c']/inputs['S_ref']
        
        # Total CD0
        outputs['CD0'] = CD0_w + CD0_c + inputs['CD_misc'] + inputs['CD_LP']
        
    def compute_partials(self, inputs, partials):
        # Wing derivatives
        partials['CD0', 'Cf_w'] = inputs['FF_w'] * inputs['Q_w'] * inputs['S_wet_w']/inputs['S_ref']
        partials['CD0', 'FF_w'] = inputs['Cf_w'] * inputs['Q_w'] * inputs['S_wet_w']/inputs['S_ref']
        partials['CD0', 'Q_w'] = inputs['Cf_w'] * inputs['FF_w'] * inputs['S_wet_w']/inputs['S_ref']
        partials['CD0', 'S_wet_w'] = inputs['Cf_w'] * inputs['FF_w'] * inputs['Q_w']/inputs['S_ref']
        
        # Canard derivatives
        partials['CD0', 'Cf_c'] = inputs['FF_c'] * inputs['Q_c'] * inputs['S_wet_c']/inputs['S_ref']
        partials['CD0', 'FF_c'] = inputs['Cf_c'] * inputs['Q_c'] * inputs['S_wet_c']/inputs['S_ref']
        partials['CD0', 'Q_c'] = inputs['Cf_c'] * inputs['FF_c'] * inputs['S_wet_c']/inputs['S_ref']
        partials['CD0', 'S_wet_c'] = inputs['Cf_c'] * inputs['FF_c'] * inputs['Q_c']/inputs['S_ref']
        
        # Reference area derivative
        S_ref = inputs['S_ref']
        partials['CD0', 'S_ref'] = -(inputs['Cf_w'] * inputs['FF_w'] * inputs['Q_w'] * inputs['S_wet_w'] +
                                    inputs['Cf_c'] * inputs['FF_c'] * inputs['Q_c'] * inputs['S_wet_c'])/(S_ref**2)
        
        # Miscellaneous and L&P derivatives
        partials['CD0', 'CD_misc'] = 1.0
        partials['CD0', 'CD_LP'] = 1.0 