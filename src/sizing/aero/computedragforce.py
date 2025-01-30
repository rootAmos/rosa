import openmdao.api as om
import numpy as np

class DragForce(om.ExplicitComponent):
    """
    Calculates the total drag force using D = CD * 0.5 * rho * V^2 * S.
    
    Inputs:
        CD : float
            Total drag coefficient [-]
        rho : float
            Air density [kg/m^3]
        V : float
            Airspeed [m/s]
        S : float
            Reference area [m^2]
    
    Outputs:
        D : float
            Drag force [N]
    """
    
    def setup(self):
        # Inputs
        self.add_input('CD', val=0.0, desc='Total drag coefficient')
        self.add_input('rho', val=1.225, units='kg/m**3', desc='Air density')
        self.add_input('V', val=0.0, units='m/s', desc='Airspeed')
        self.add_input('S', val=0.0, units='m**2', desc='Reference area')
        
        # Output
        self.add_output('D', val=0.0, units='N', desc='Drag force')
        
        # Declare partials
        self.declare_partials('D', ['CD', 'rho', 'V', 'S'])
        
    def compute(self, inputs, outputs):
        CD = inputs['CD']
        rho = inputs['rho']
        V = inputs['V']
        S = inputs['S']
        
        # Compute dynamic pressure
        q = 0.5 * rho * V**2
        
        # Compute drag force
        outputs['D'] = CD * q * S
        
    def compute_partials(self, inputs, partials):
        CD = inputs['CD']
        rho = inputs['rho']
        V = inputs['V']
        S = inputs['S']
        
        # Partial derivatives
        partials['D', 'CD'] = 0.5 * rho * V**2 * S
        partials['D', 'rho'] = 0.5 * CD * V**2 * S
        partials['D', 'V'] = CD * rho * V * S
        partials['D', 'S'] = 0.5 * CD * rho * V**2 