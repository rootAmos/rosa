import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('../../../'))))

from src.sizing.aero.lift.group_cl0 import GroupCL0



class GroupCL0MantaRay(om.Group):

    """
    Group that computes total zero-angle lift coefficient by combining wing and canard.

    Includes computation chain from airfoil properties through to CL0.
    """
    
    def initialize(self):
        self.options.declare('d_alpha0_d_twist', default=-0.405, types=float)
        self.options.declare('N', default=1, desc='Number of nodes')

    def setup(self):

        N = self.options['N']

        self.add_subsystem('manta_cl0',
                        GroupCL0(N=N),
                        promotes_inputs=[],
                        promotes_outputs=[])
        

        self.add_subsystem('ray_cl0',
                        GroupCL0(N=N),
                        promotes_inputs=[],
                        promotes_outputs=[])


        
        adder = om.AddSubtractComp()
        adder.add_equation('CL0_total',
                        ['CL0_manta', 'CL0_ray'],
                        desc='Total zero-angle lift coefficient',
                        vec_size=N)


        self.add_subsystem('sum_CL0', adder, promotes=['*'])


        self.connect('manta_cl0.CL0', 'CL0_manta')
        self.connect('ray_cl0.CL0', 'CL0_ray')

        # end


if __name__ == "__main__":

    # Create problem instance
    prob = om.Problem(reports=False)
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    N = 1
    
    # Wing parameters
    ivc.add_output('alpha0_airfoil_manta', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_twist_manta', val=0.0, units='rad', desc='twist angle')
    ivc.add_output('CL_alpha_manta_eff', val=5.0 * np.ones(N), units='1/rad', desc='Wing lift curve slope')

    # Ray parameters
    ivc.add_output('alpha0_airfoil_ray', val=-1.5, units='deg', desc='Canard airfoil zero-lift angle')
    ivc.add_output('d_twist_ray', val=0.0, units='rad', desc='twist angle')
    ivc.add_output('CL_alpha_ray_eff', val=4.0 * np.ones(N), units='1/rad', desc='Canard lift curve slope')


    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('CL0', GroupCL0MantaRay(N=N), promotes=['*'])


    prob.model.connect('alpha0_airfoil_manta', 'manta_cl0.alpha0_airfoil')
    prob.model.connect('d_twist_manta', 'manta_cl0.d_twist')
    prob.model.connect('CL_alpha_manta_eff', 'manta_cl0.CL_alpha_eff')


    prob.model.connect('alpha0_airfoil_ray', 'ray_cl0.alpha0_airfoil')
    prob.model.connect('CL_alpha_ray_eff', 'ray_cl0.CL_alpha_eff')
    prob.model.connect('d_twist_ray', 'ray_cl0.d_twist')



    # Setup problem
    prob.setup()

    #om.n2(prob.model)
    
    # Run baseline case
    prob.run_model()


    print('\nBaseline Configuration:')
    print('----------------------')
    print('Wing:')
    print(f'  Alpha_0:             {np.degrees(prob.get_val("manta_cl0.alpha0")[0]):8.3f} deg')
    print(f'  CL_0:                {prob.get_val("CL0_manta")[0]:8.3f}')


    print('\nCanard:')
    print(f'  Alpha_0:             {np.degrees(prob.get_val("ray_cl0.alpha0")[0]):8.3f} deg')
    print(f'  CL_0:                {prob.get_val("CL0_ray")[0]:8.3f}')


    print('\nTotal:')
    print(f'  CL_0:                {prob.get_val("CL0_total")[0]:8.3f}')
    
# Parameter sweeps

    # Wing zero-lift angle sweep
    alpha0_range = np.linspace(-4.0, 0.0, 50)
    CL0_manta = []
    CL0_ray = []
    CL0_total = []
    for a0 in alpha0_range:
        prob.set_val('manta_cl0.alpha0_airfoil', a0)
        prob.run_model()

        CL0_manta.append(prob.get_val('CL0_manta')[0])
        CL0_ray.append(prob.get_val('CL0_ray')[0])
        CL0_total.append(prob.get_val('CL0_total')[0])




    plt.plot(alpha0_range, CL0_manta, label='Manta')
    plt.plot(alpha0_range, CL0_ray, '--', label='Ray')
    plt.plot(alpha0_range, CL0_total, '--', label='Total')
    plt.xlabel('Wing Zero-Lift Angle [deg]')
    plt.ylabel('$CL_{0}$')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Wing Zero-Lift Angle')


    plt.show()


    # Mach number sweep

    prob.check_partials(compact_print=True) 