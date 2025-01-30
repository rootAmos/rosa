import openmdao.api as om

class LeakageDrag(om.ExplicitComponent):
    """
    Outputs a constant leakage drag coefficient.
    """
    
    def initialize(self):
        self.options.declare('CD0_leakage', default=0.0003,
                           desc='Leakage drag coefficient')
    
    def setup(self):
        self.add_output('CD0_leakage',
                       val=self.options['CD0_leakage'],
                       desc='Leakage drag coefficient')
        
        # No partials to declare since there are no inputs


class ExcrescenceDrag(om.ExplicitComponent):
    """
    Outputs a constant excrescence drag coefficient.
    """
    
    def initialize(self):
        self.options.declare('CD0_excrescence', default=0.0002,
                           desc='Excrescence drag coefficient')
    
    def setup(self):
        self.add_output('CD0_excrescence',
                       val=self.options['CD0_excrescence'],
                       desc='Excrescence drag coefficient')
        
        # No partials to declare since there are no inputs 