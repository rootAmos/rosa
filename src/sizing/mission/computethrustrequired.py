import openmdao.api as om
import numpy as np

class ThrustRequired(om.ExplicitComponent):
    """
    Calculates the required thrust based on drag, weight, flight path angle, and acceleration.
    T = D + W*sin(gamma) + m*udot
    
    Inputs:
        D : float
            Drag force [N]
        W : float
            Aircraft weight [N]
        gamma : float
            Flight path angle [rad]
        mass : float
            Aircraft mass [kg]
        udot : float
            Acceleration along flight path [m/s^2]
    
    Outputs:
        T_req : float
            Required thrust [N]
    """
    
    def setup(self):
        # Inputs
        self.add_input('D', val=1.0, units='N', desc='Drag force')
        self.add_input('W', val=1.0, units='N', desc='Aircraft weight')
        self.add_input('gamma', val=1.0, units='rad', desc='Flight path angle')
        self.add_input('mass', val=1.0, units='kg', desc='Aircraft mass')

        self.add_input('udot', val=1.0, units='m/s**2', desc='Acceleration')
        

        # Output
        self.add_output('T_req', val=1.0, units='N', desc='Required thrust')
        

        # Declare partials
        self.declare_partials('T_req', ['D', 'W', 'gamma', 'mass', 'udot'])
        
    def compute(self, inputs, outputs):
        D = inputs['D']
        W = inputs['W']
        gamma = inputs['gamma']
        mass = inputs['mass']
        udot = inputs['udot']
        
        # Compute required thrust
        outputs['T_req'] = D + W * np.sin(gamma) + mass * udot
        
    def compute_partials(self, inputs, partials):
        W = inputs['W']
        gamma = inputs['gamma']
        
        # Partial derivatives
        partials['T_req', 'D'] = 1.0
        partials['T_req', 'W'] = np.sin(gamma)
        partials['T_req', 'gamma'] = W * np.cos(gamma)
        partials['T_req', 'mass'] = inputs['udot']
        partials['T_req', 'udot'] = inputs['mass'] 