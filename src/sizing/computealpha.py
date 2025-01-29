import openmdao.api as om
import numpy as np

class AngleOfAttack(om.ImplicitComponent):
    """
    Solves for the angle of attack by matching required lift to actual lift.
    
    Inputs:
        weight : float
            Aircraft weight [N]
        load_factor : float
            Load factor [-]
        L : float
            Current total lift force [N]
    
    Outputs:
        alpha : float
            Angle of attack [rad]
    
    Residuals:
        R_alpha : float
            Lift equilibrium residual [N]
    """
    
    def setup(self):
        # Inputs
        self.add_input('weight', val=0.0, units='N', desc='Aircraft weight')
        self.add_input('load_factor', val=1.0, desc='Load factor')
        self.add_input('L', val=0.0, units='N', desc='Current lift force')
        
        # Output state
        self.add_output('alpha', val=0.0, units='rad', 
                       desc='Angle of attack',
                       lower=-np.pi/6,  # -30 degrees
                       upper=np.pi/6)   # +30 degrees
        
        # Declare partials
        self.declare_partials('alpha', ['weight', 'load_factor', 'L', 'alpha'])
        
    def apply_nonlinear(self, inputs, outputs, residuals):
        """Compute residual for angle of attack."""
        weight = inputs['weight']
        n = inputs['load_factor']
        L = inputs['L']
        
        # Residual is difference between required and actual lift
        residuals['alpha'] = L - weight * n
        
    def linearize(self, inputs, outputs, partials):
        """Compute partial derivatives analytically."""
        # Partial derivatives of residual with respect to inputs
        partials['alpha', 'weight'] = -inputs['load_factor']
        partials['alpha', 'load_factor'] = -inputs['weight']
        partials['alpha', 'L'] = 1.0
        
        # Partial derivative with respect to state variable (alpha)
        # This will be computed by the total derivative calculation
        # based on connections to other components
        partials['alpha', 'alpha'] = 0.0 