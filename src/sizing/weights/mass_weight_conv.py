
import openmdao.api as om

class MassToWeight(om.ExplicitComponent):
    """


    Computes weight (force) from mass using W = mg
    """
    
    def initialize(self):
        self.options.declare('g', default=9.806, desc='Gravitational acceleration (m/s^2)')
        
    def setup(self):
        # Inputs
        self.add_input('mass', val=1.0, units='kg',
                      desc='Mass of object')
        
        # Outputs
        self.add_output('weight', val=1.0, units='N',
                      desc='Weight (force) of object')
        
        # Declare partials
        self.declare_partials('weight', 'mass')
        
    def compute(self, inputs, outputs):
        g = self.options['g']
        
        outputs['weight'] = inputs['mass'] * g
        
    def compute_partials(self, inputs, partials):
        g = self.options['g']
        
        partials['weight', 'mass'] = g


class WeightToMass(om.ExplicitComponent):
    """
    Computes mass from weight (force) using m = W/g
    """
    
    def initialize(self):
        self.options.declare('g', default=9.806, desc='Gravitational acceleration (m/s^2)')
        
    def setup(self):
        # Inputs
        self.add_input('weight', val=1.0, units='N',
                      desc='Weight (force) of object')
        
        # Outputs
        self.add_output('mass', val=1.0, units='kg',
                      desc='Mass of object')
        
        # Declare partials
        self.declare_partials('mass', 'weight')
        
    def compute(self, inputs, outputs):
        g = self.options['g']
        
        outputs['mass'] = inputs['weight'] / g
        
    def compute_partials(self, inputs, partials):
        g = self.options['g']
        
        partials['mass', 'weight'] = 1.0 / g