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
        self.add_input('lift_req', val=1.0, desc='lift force required')
        self.add_input('lift_calc', val=0.0, units='N', desc='lift force calculated')
        
        # Output state
        self.add_output('alpha', val=0.0, units='rad', 
                       desc='Angle of attack',
                       lower=-np.pi/6,  # -30 degrees
                       upper=np.pi/6)   # +30 degrees
        
        # Declare partials
        self.declare_partials('alpha', ['lift_req', 'lift_calc', 'alpha'])
        
    def apply_nonlinear(self, inputs, outputs, residuals):
        """Compute residual for angle of attack."""
        lift_req = inputs['lift_req']
        lift_calc = inputs['lift_calc']
        
        # Residual is difference between required and actual lift
        residuals['alpha'] = lift_req - lift_calc
        
    def linearize(self, inputs, outputs, partials):
        """Compute partial derivatives analytically."""
        # Partial derivatives of residual with respect to inputs
        partials['alpha', 'lift_req'] = 1.0
        partials['alpha', 'lift_calc'] = -1.0
        
        # Partial derivative with respect to state variable (alpha)
        # This will be computed by the total derivative calculation
        # based on connections to other components
        partials['alpha', 'alpha'] = 0.0 