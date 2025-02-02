import openmdao.api as om
from src.sizing.aero.drag.cd0 import ZeroLiftDragComponent
from misc_cd0 import LeakageDrag, ExcrescenceDrag
from group_prplsr_cd0 import  PodDragGroup

import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from mission.computeamos import ComputeAtmos
from mission.mach_number import MachNumber
from reynolds import ReynoldsNumber
from skin_friction import SkinFriction
from geometry.wing_form_factor import FormFactor
from geometry.wing_wetted import WettedAreaWing








class GroupCD0Wing(om.Group):
    """
    Group that combines zero-lift drag components for Ray configuration.
    """
    def initialize(self):
        pass

    def setup(self):


        self.add_subsystem('reynolds', ReynoldsNumber(),
                          promotes_inputs=['*'],promotes_outputs=['*'])


        self.add_subsystem('wet', WettedAreaWing(),
                          promotes_inputs=['*'],promotes_outputs=['*'])
        

        self.add_subsystem('ff', FormFactor(),
                          promotes_inputs=['*'],promotes_outputs=['*'])
        


        self.add_subsystem('cf', SkinFriction(),
                          promotes_inputs=['*'],promotes_outputs=['*'])
    

        self.add_subsystem('cd0', ZeroLiftDragComponent(),
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
    ivc.add_output('lambda_w', val=0.45, desc='Wing taper ratio')
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