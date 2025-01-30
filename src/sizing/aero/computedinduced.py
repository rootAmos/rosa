import openmdao.api as om
import numpy as np

class InducedDrag(om.ExplicitComponent):
    """
    Calculates induced drag coefficient for a single lifting surface.
    
    Inputs:
        CL : float
            Lift coefficient [-]
        AR : float
            Aspect ratio [-]
        e : float
            Oswald efficiency factor [-]
    
    Outputs:
        CDi : float
            Induced drag coefficient [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('CL', val=0.0, desc='Lift coefficient')
        self.add_input('AR', val=0.0, desc='Aspect ratio')
        self.add_input('e', val=0.85, desc='Oswald efficiency factor')
        
        # Output
        self.add_output('CDi', val=0.0, desc='Induced drag coefficient')
        
        # Declare partials
        self.declare_partials('CDi', ['CL', 'AR', 'e'])
        
    def compute(self, inputs, outputs):
        CL = inputs['CL']
        AR = inputs['AR']
        e = inputs['e']
        
        # Compute induced drag coefficient
        outputs['CDi'] = CL**2 / (np.pi * AR * e)
        
    def compute_partials(self, inputs, partials):
        CL = inputs['CL']
        AR = inputs['AR']
        e = inputs['e']
        
        # Derivatives
        partials['CDi', 'CL'] = 2 * CL / (np.pi * AR * e)
        partials['CDi', 'AR'] = -CL**2 / (np.pi * AR**2 * e)
        partials['CDi', 'e'] = -CL**2 / (np.pi * AR * e**2) 