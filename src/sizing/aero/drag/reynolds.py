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

    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')


    def setup(self):
        # Inputs

        N = self.options['N']


        self.add_input('rho', val=1.0 * np.ones(N), units='kg/m**3', 
                      desc='Fluid density')
        self.add_input('u', val=1.0 * np.ones(N), units='m/s', 
                      desc='Flow speed')

        self.add_input('l_char', val=1.0 , units='m', 
                      desc='Characteristic length')
        self.add_input('mu', val=1 * np.ones(N), units='Pa*s', 
                      desc='Dynamic viscosity')
        


        # Outputs
        self.add_output('Re', val=0.0 * np.ones(N), desc='Reynolds number')
        
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

        N = self.options['N']

        # Partial derivatives
        partials['Re', 'rho'] = np.eye(N) * (u * l_char) / mu
        partials['Re', 'u'] = np.eye(N) * (rho * l_char) / mu
        partials['Re', 'l_char'] = (rho * u) / mu
        partials['Re', 'mu'] = np.eye(N) * -(rho * u * l_char) / (mu**2) 