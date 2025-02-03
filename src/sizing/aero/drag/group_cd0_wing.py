import openmdao.api as om


import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


from src.sizing.aero.drag.reynolds import ReynoldsNumber
from src.sizing.aero.drag.skin_friction import SkinFriction
from src.sizing.geometry.wing_form_factor import FormFactor
from src.sizing.geometry.wing_wetted import WettedAreaWing
from src.sizing.aero.drag.cd0 import ZeroLiftDragComponent







class GroupCD0Wing(om.Group):
    """
    Group that combines zero-lift drag components for Ray configuration.
    """
    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')

    def setup(self):

        N = self.options['N']



        self.add_subsystem('reynolds', ReynoldsNumber(N=N),
                          promotes_inputs=['*'],promotes_outputs=['*'])



        self.add_subsystem('wet', WettedAreaWing(),
                          promotes_inputs=['*'],promotes_outputs=['*'])
        

        self.add_subsystem('ff', FormFactor(N=N),
                          promotes_inputs=['*'],promotes_outputs=['*'])
        


        self.add_subsystem('cf', SkinFriction(N=N),
                          promotes_inputs=['*'],promotes_outputs=['*'])
    


        self.add_subsystem('cd0', ZeroLiftDragComponent(N=N),
                        promotes_inputs=['*'],promotes_outputs=['*'])


    
        self.connect('ff_wing', 'ff')


if __name__ == "__main__":

    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp for inputs
    ivc = om.IndepVarComp()
    ivc.add_output('Q', val=1.1, desc='Interference factor')
    ivc.add_output('S_ref', val=50.0, units='m**2', desc='Reference area')
    ivc.add_output('S_exp', val=50.0, units='m**2', desc='Exposed planform area')
    ivc.add_output('t_c', val=0.19, desc='Thickness to chord ratio')
    ivc.add_output('tau', val=0.8, desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
    ivc.add_output('lambda', val=0.45, desc='Wing taper ratio')
    ivc.add_output('k_lam', val=0.1, desc='Laminar flow fraction')
    ivc.add_output('sweep_max_t', val=10,units='deg', desc='Wing sweep at maximum thickness')


    #ivc.add_output('alt', val=30000, units='ft', desc='Altitude')
    ivc.add_output('u', val=50.0, units='m/s', desc='Flow speed')
    ivc.add_output('mach', val=0.8, desc='Mach number')

    ivc.add_output('rho', val=0.3639, units='kg/m**3', desc='Density')


    ivc.add_output('l_char', val=1.0, units='m',desc='Characteristic length')
    ivc.add_output('mu', val=1.789e-5, units='Pa*s', desc='Dynamic viscosity')
    

    # Add IVC and group to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cd0_group', GroupCD0Wing(), promotes=['*'])
    


    # Setup and run problem
    prob.setup()
    om.n2(prob)
    prob.run_model()
    


    # Print results

    print('\nComponent CD0s:')
    print(f'  Wing CD0 = {prob.get_val("cd0_group.cd0_wing.CD0"):.6f}')
    print(f'  Canard CD0 = {prob.get_val("cd0_group.cd0_canard.CD0"):.6f}')
    print(f'  Total CD0 = {prob.get_val("cd0_group.cd0_total.CD0_total"):.6f}')
    
    # Check partials
    prob.check_partials(compact_print=True)