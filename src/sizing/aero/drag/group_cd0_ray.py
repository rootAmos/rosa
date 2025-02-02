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
from group_cd0_wing import GroupCD0Wing
from group_prplsr_cd0 import PodDragGroup


class GroupCD0Ray(om.Group):
    """
    Group that combines zero-lift drag components for Ray configuration.
    """
    def initialize(self):
        pass

    def setup(self):

        
        self.add_subsystem('canard', GroupCD0Wing(),
                          promotes_inputs=['mach','rho','u','mu','sweep_max_t','t_c','tau','lambda_w','S_ref','S_exp'],
                          promotes_outputs=[])
        
        self.add_subsystem('pods', PodDragGroup(),
                          promotes_inputs=['mach','rho','u','mu','num_pods','S_ref','l_pod','d_pod'],
                          promotes_outputs=[])

        
        adder = om.AddSubtractComp()
        adder.add_equation('CD0',
                        ['canard_CD0', 'pods_CD0'],
                        desc='Total drag coefficient')

        self.add_subsystem('sum', adder, promotes=[])

        self.connect('canard.CD0', 'sum.canard_CD0')
        self.connect('pods.CD0_total_pods', 'sum.pods_CD0')





if __name__ == "__main__":



    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp for inputs
    ivc = om.IndepVarComp()
    ivc.add_output('mach', val=0.8, desc='Mach number')
    ivc.add_output('rho', val=0.3639, units='kg/m**3', desc='Density')
    ivc.add_output('Q_ray', val=1.1, desc='Interference factor')
    ivc.add_output('S_exp', val=50.0, units='m**2', desc='Exposed planform area')
    ivc.add_output('t_c', val=0.19, desc='Thickness to chord ratio')

    ivc.add_output('tau', val=0.8, desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
    ivc.add_output('lambda_w', val=0.45, desc='Wing taper ratio')
    ivc.add_output('k_lam_ray', val=0.1, desc='Laminar flow fraction')
    ivc.add_output('sweep_max_t', val=10,units='deg', desc='Wing sweep at maximum thickness')
    ivc.add_output('l_char_ray', val=1.0, units='m',desc='Characteristic length')

    ivc.add_output('S_ref', val=100.0, units='m**2', desc='Reference area')



    ivc.add_output('alt', val=30000, units='ft', desc='Altitude')
    ivc.add_output('u', val=50.0, units='m/s', desc='Flow speed')
    ivc.add_output('mu', val=1.789e-5, units='Pa*s', desc='Dynamic viscosity')

    ivc.add_output('l_pod', val=3.0, units='m', desc='Pod length')
    ivc.add_output('d_pod', val=2.0, units='m', desc='Pod diameter')

    ivc.add_output('Q_pod', val=1.0, desc='Pod interference factor')
    ivc.add_output('k_lam_pod', val=0.1, desc='Pod laminar flow fraction')
    ivc.add_output('num_pods', val=2, desc='Number of pods')



    # Add IVC and group to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cd0_ray', GroupCD0Ray(), promotes=['*'])


    prob.model.connect('Q_ray', 'canard.Q')
    prob.model.connect('Q_pod', 'pods.Q')

    prob.model.connect('l_char_ray', 'canard.l_char')
    prob.model.connect('l_pod', 'pods.l_char')

    prob.model.connect('k_lam_ray', 'canard.k_lam')
    prob.model.connect('k_lam_pod', 'pods.k_lam')





    



    # Setup and run problem


    prob.setup()
    #som.n2(prob)
    prob.run_model()
    


    # Print results

    print('\nComponent CD0s:')
    print(f'  Canard CD0 = {prob.get_val("canard.CD0")}')
    print(f'  Pods CD0 = {prob.get_val("pods.CD0_total_pods")}')
    print(f'  Total CD0 = {prob.get_val("sum.CD0")}')
    

    # Check partials
    prob.check_partials(compact_print=True)