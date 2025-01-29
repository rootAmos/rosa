import openmdao.api as om
import numpy as np

class ReynoldsNumber(om.ExplicitComponent):
    """
    Calculates Reynolds number for a fluid flow.
    
    Inputs:
        rho : float
            Density of the fluid [kg/m^3]
        V : float
            Flow speed [m/s]
        L : float
            Characteristic length [m]
        mu : float
            Dynamic viscosity [kg/(m*s)]
    
    Outputs:
        Re : float
            Reynolds number [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('rho', val=1.225, units='kg/m**3', 
                      desc='Fluid density')
        self.add_input('V', val=0.0, units='m/s', 
                      desc='Flow speed')
        self.add_input('L', val=1.0, units='m', 
                      desc='Characteristic length')
        self.add_input('mu', val=1.789e-5, units='kg/(m*s)', 
                      desc='Dynamic viscosity')
        
        # Outputs
        self.add_output('Re', val=0.0, desc='Reynolds number')
        
        # Declare partials
        self.declare_partials('Re', ['rho', 'V', 'L', 'mu'])
        
    def compute(self, inputs, outputs):
        rho = inputs['rho']
        V = inputs['V']
        L = inputs['L']
        mu = inputs['mu']
        
        # Reynolds number equation from image
        outputs['Re'] = (rho * V * L) / mu
        
    def compute_partials(self, inputs, partials):
        rho = inputs['rho']
        V = inputs['V']
        L = inputs['L']
        mu = inputs['mu']
        
        # Partial derivatives
        partials['Re', 'rho'] = (V * L) / mu
        partials['Re', 'V'] = (rho * L) / mu
        partials['Re', 'L'] = (rho * V) / mu
        partials['Re', 'mu'] = -(rho * V * L) / (mu**2) 