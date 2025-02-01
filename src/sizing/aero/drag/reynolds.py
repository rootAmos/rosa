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
        self.add_input('rho', val=1.0, units='kg/m**3', 
                      desc='Fluid density')
        self.add_input('u', val=0.0, units='m/s', 
                      desc='Flow speed')
        self.add_input('l_char', val=1.0, units='m', 
                      desc='Characteristic length')
        self.add_input('mu', val=1.789e-5, units='Pa*s', 
                      desc='Dynamic viscosity')
        

        # Outputs
        self.add_output('Re', val=0.0, desc='Reynolds number')
        
        # Declare partials
        self.declare_partials('Re', ['rho', 'u', 'l_char', 'mu'])
        
    def compute(self, inputs, outputs):


        rho = inputs['rho']
        u = inputs['u']
        l_char = inputs['l_char']
        mu = inputs['mu']
        

        # Reynolds number equation from image
        outputs['Re'] = (rho * u * l_char) / mu
        

    def compute_partials(self, inputs, partials):
        
        rho = inputs['rho']
        u = inputs['u']
        l_char = inputs['l_char']
        mu = inputs['mu']

        # Partial derivatives
        partials['Re', 'rho'] = (u * l_char) / mu
        partials['Re', 'u'] = (rho * l_char) / mu
        partials['Re', 'l_char'] = (rho * u) / mu
        partials['Re', 'mu'] = -(rho * u * l_char) / (mu**2) 