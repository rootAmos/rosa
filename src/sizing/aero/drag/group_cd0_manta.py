import openmdao.api as om
from src.sizing.aero.drag.cd0 import ZeroLiftDragComponent
from misc_cd0 import LeakageDrag, ExcrescenceDrag
from group_prplsr_cd0 import  PodDragGroup

import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.sizing.mission.atmos import ComputeAtmos
from mission.mach_number import MachNumber
from group_cd0_wing import GroupCD0Wing
from group_prplsr_cd0 import DuctDragGroup


class GroupCD0Manta(om.Group):
    """
    Group that combines zero-lift drag components for Ray configuration.
    """
    def initialize(self):
        pass

    def setup(self):

        
        self.add_subsystem('wing', GroupCD0Wing(),
                          promotes_inputs=['mach','rho','u','mu','sweep_max_t','t_c','tau','lambda_w','S_ref','S_exp'],
                          promotes_outputs=[])
        
        self.add_subsystem('ducts', DuctDragGroup(),
                          promotes_inputs=['mach','rho','u','mu','num_ducts','S_ref','c_duct','od_duct','id_duct'],
                          promotes_outputs=[])


        
        adder = om.AddSubtractComp()
        adder.add_equation('CD0',
                        ['manta_CD0', 'ducts_CD0'],
                        desc='Total drag coefficient')

        self.add_subsystem('sum', adder, promotes=[])

        self.connect('wing.CD0', 'sum.manta_CD0')
        self.connect('ducts.CD0_total_ducts', 'sum.ducts_CD0')





if __name__ == "__main__":



    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp for inputs
    ivc = om.IndepVarComp()

    ivc.add_output('mach', val=0.8, desc='Mach number')
    ivc.add_output('rho', val=0.3639, units='kg/m**3', desc='Density')

    ivc.add_output('Q_manta', val=1.1, desc='Interference factor')
    ivc.add_output('S_exp', val=50.0, units='m**2', desc='Exposed planform area')
    ivc.add_output('t_c', val=0.19, desc='Thickness to chord ratio')

    ivc.add_output('tau', val=0.8, desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
    ivc.add_output('lambda_w', val=0.45, desc='Wing taper ratio')
    ivc.add_output('k_lam_manta', val=0.1, desc='Laminar flow fraction')
    ivc.add_output('sweep_max_t', val=10,units='deg', desc='Wing sweep at maximum thickness')
    ivc.add_output('l_char_manta', val=1.0, units='m',desc='Characteristic length')

    ivc.add_output('S_ref', val=100.0, units='m**2', desc='Reference area')

    ivc.add_output('alt', val=30000, units='ft', desc='Altitude')
    ivc.add_output('u', val=50.0, units='m/s', desc='Flow speed')
    ivc.add_output('mu', val=1.789e-5, units='Pa*s', desc='Dynamic viscosity')

    ivc.add_output('c_duct', val=3.0, units='m', desc='Nacelle length')
    ivc.add_output('od_duct', val=2.0, units='m', desc='Duct outer diameter')
    ivc.add_output('id_duct', val=1.0, units='m', desc='Duct inner diameter')
    ivc.add_output('Q_duct', val=1.0, desc='Duct interference factor')
    ivc.add_output('k_lam_duct', val=0.1, desc='Duct laminar flow fraction')
    ivc.add_output('num_ducts', val=2, desc='Number of ducts')



    # Add IVC and group to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cd0_manta', GroupCD0Manta(), promotes=['*'])


    prob.model.connect('Q_manta', 'wing.Q')
    prob.model.connect('Q_duct', 'ducts.Q')


    prob.model.connect('l_char_manta', 'wing.l_char')
    prob.model.connect('c_duct', 'ducts.l_char')

    prob.model.connect('k_lam_manta', 'wing.k_lam')
    prob.model.connect('k_lam_duct', 'ducts.k_lam')





    



    # Setup and run problem


    prob.setup()
    #som.n2(prob)
    prob.run_model()
    


    # Print results

    print('\nComponent CD0s:')
    print(f'  Wing CD0 = {prob.get_val("wing.CD0")}')
    print(f'  Ducts CD0 = {prob.get_val("ducts.CD0_total_ducts")}')
    print(f'  Total CD0 = {prob.get_val("sum.CD0")}')
    


    # Check partials
    prob.check_partials(compact_print=True)