import openmdao.api as om
from src.sizing.aero.drag.cd0 import ZeroLiftDragComponent
from misc_cd0 import LeakageDrag, ExcrescenceDrag
from group_prplsr_cd0 import  PodDragGroup

import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from group_cd0_manta import GroupCD0Manta
from group_cd0_ray import GroupCD0Ray


class GroupCD0MantaRay(om.Group):
    """
    Group that combines zero-lift drag components for Ray configuration.
    """
    def initialize(self):
        pass

    def setup(self):
        
        self.add_subsystem('manta', GroupCD0Manta(),
                          promotes_inputs=['mach','rho','u','mu','S_ref'],
                          promotes_outputs=[])
        
        self.add_subsystem('ray', GroupCD0Ray(),
                          promotes_inputs=['mach','rho','u','mu','S_ref'],
                          promotes_outputs=[])

        
        adder = om.AddSubtractComp()
        adder.add_equation('CD0',
                        ['manta_CD0', 'ray_CD0'],
                        desc='Total drag coefficient')


        self.add_subsystem('sum', adder, promotes=[])


        self.connect('manta.sum.CD0', 'sum.manta_CD0')
        self.connect('ray.sum.CD0', 'sum.ray_CD0')







if __name__ == "__main__":



    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp for inputs
    ivc = om.IndepVarComp()


    ivc.add_output('Q_manta', val=1.1, desc='Interference factor')
    ivc.add_output('S_exp_manta', val=50.0, units='m**2', desc='Exposed planform area')
    ivc.add_output('t_c_manta', val=0.19, desc='Thickness to chord ratio')

    ivc.add_output('tau_manta', val=0.8, desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
    ivc.add_output('lambda_manta', val=0.45, desc='Wing taper ratio')
    ivc.add_output('k_lam_manta', val=0.1, desc='Laminar flow fraction')

    ivc.add_output('sweep_max_t_manta', val=10,units='deg', desc='Wing sweep at maximum thickness')
    ivc.add_output('l_char_manta', val=1.0, units='m',desc='Characteristic length')


    ivc.add_output('Q_ray', val=1.1, desc='Interference factor')
    ivc.add_output('S_exp_ray', val=50.0, units='m**2', desc='Exposed planform area')
    ivc.add_output('t_c_ray', val=0.19, desc='Thickness to chord ratio')


    ivc.add_output('tau_ray', val=0.8, desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
    ivc.add_output('lambda_ray', val=0.45, desc='Wing taper ratio')
    ivc.add_output('k_lam_ray', val=0.1, desc='Laminar flow fraction')

    ivc.add_output('sweep_max_t_ray', val=10,units='deg', desc='Wing sweep at maximum thickness')
    ivc.add_output('l_char_ray', val=1.0, units='m',desc='Characteristic length')




    ivc.add_output('S_ref', val=100.0, units='m**2', desc='Reference area')

    ivc.add_output('u', val=50.0, units='m/s', desc='Flow speed')
    ivc.add_output('mu', val=1.789e-5, units='Pa*s', desc='Dynamic viscosity')
    ivc.add_output('rho', val=0.3639, units='kg/m**3', desc='Density')
    ivc.add_output('mach', val=50/300, desc='Mach number')

    ivc.add_output('c_duct', val=3.0, units='m', desc='Nacelle length')
    ivc.add_output('od_duct', val=2.0, units='m', desc='Duct outer diameter')
    ivc.add_output('id_duct', val=1.0, units='m', desc='Duct inner diameter')
    ivc.add_output('Q_duct', val=1.0, desc='Duct interference factor')
    ivc.add_output('k_lam_duct', val=0.1, desc='Duct laminar flow fraction')
    ivc.add_output('num_ducts', val=2, desc='Number of ducts')


    ivc.add_output('l_pod', val=3.0, units='m', desc='Pod length')
    ivc.add_output('d_pod', val=2.0, units='m', desc='Pod diameter')

    ivc.add_output('Q_pod', val=1.0, desc='Pod interference factor')
    ivc.add_output('k_lam_pod', val=0.1, desc='Pod laminar flow fraction')
    ivc.add_output('num_pods', val=2, desc='Number of pods')



    # Add IVC and group to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cd0_manta_ray', GroupCD0MantaRay(), promotes=['*'])



    prob.model.connect('Q_manta', 'manta.wing.Q')
    prob.model.connect('Q_duct', 'manta.ducts.Q')
    prob.model.connect('Q_ray', 'ray.canard.Q')
    prob.model.connect('Q_pod', 'ray.pods.Q')




    prob.model.connect('l_char_manta', 'manta.wing.l_char')
    prob.model.connect('c_duct', 'manta.ducts.l_char')
    prob.model.connect('l_char_ray', 'ray.canard.l_char')
    prob.model.connect('l_pod', ['ray.pods.l_char','ray.l_pod'])
    prob.model.connect('d_pod', ['ray.d_pod'])






    prob.model.connect('k_lam_manta', 'manta.wing.k_lam')
    prob.model.connect('k_lam_duct', 'manta.ducts.k_lam')
    prob.model.connect('k_lam_ray', 'ray.canard.k_lam')
    prob.model.connect('k_lam_pod', 'ray.pods.k_lam')


    prob.model.connect('sweep_max_t_manta', 'manta.sweep_max_t')
    prob.model.connect('sweep_max_t_ray', 'ray.sweep_max_t')



    prob.model.connect('S_exp_manta', 'manta.S_exp')
    prob.model.connect('S_exp_ray', 'ray.S_exp')



    prob.model.connect('t_c_manta', 'manta.t_c')
    prob.model.connect('t_c_ray', 'ray.t_c')



    prob.model.connect('tau_manta', 'manta.tau')
    prob.model.connect('tau_ray', 'ray.tau')    


    prob.model.connect('lambda_manta', 'manta.lambda_w')
    prob.model.connect('lambda_ray', 'ray.lambda_w')


    prob.model.connect('num_ducts', 'manta.num_ducts')
    prob.model.connect('num_pods', 'ray.num_pods')


    prob.model.connect('od_duct', 'manta.od_duct')
    prob.model.connect('id_duct', 'manta.id_duct')
    prob.model.connect('c_duct', 'manta.c_duct')




    


    


    








    # Setup and run problem




    prob.setup()
    om.n2(prob)
    prob.run_model()
    


    # Print results

    print('\nComponent CD0s:')
    print(f'  Manta CD0 = {prob.get_val("manta.sum.CD0")}')
    print(f'  Ray CD0 = {prob.get_val("ray.sum.CD0")}')
    print(f'  Total CD0 = {prob.get_val("sum.CD0")}')
    

    # Check partials
    prob.check_partials(compact_print=True)