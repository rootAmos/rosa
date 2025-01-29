import openmdao.api as om

class ZeroLiftDragGroup(om.Group):
    """
    Group that combines zero-lift drag from multiple components and adds miscellaneous drag.
    """
    
    def setup(self):
        # Add components for wing and canard
        self.add_subsystem('cd0_wing', 
                          ZeroLiftDragComponent(),
                          promotes_inputs=[('S_ref', 'S_ref')])
        
        self.add_subsystem('cd0_canard', 
                          ZeroLiftDragComponent(),
                          promotes_inputs=[('S_ref', 'S_ref')])
        
        # Add miscellaneous and L&P drag inputs
        self.add_subsystem('cd0_total', om.ExecComp(
            'CD0 = CD0_w + CD0_c + CD_misc + CD_LP',
            CD0={'val': 0.0, 'desc': 'Total zero-lift drag coefficient'},
            CD0_w={'val': 0.0, 'desc': 'Wing zero-lift drag coefficient'},
            CD0_c={'val': 0.0, 'desc': 'Canard zero-lift drag coefficient'},
            CD_misc={'val': 0.0, 'desc': 'Miscellaneous drag coefficient'},
            CD_LP={'val': 0.0, 'desc': 'Leakage and protuberance drag coefficient'}
        ))
        
    def configure(self):
        # Connect individual CD0s to total
        self.connect('cd0_wing.CD0', 'cd0_total.CD0_w')
        self.connect('cd0_canard.CD0', 'cd0_total.CD0_c') 