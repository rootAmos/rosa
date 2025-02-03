import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt


import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.sizing.aero.lift.zero_ang_lift import ZeroLiftAngle, ZeroAngleLift


class GroupCL0(om.Group):

    """
    Group that computes total zero-angle lift coefficient by combining wing and canard.

    Includes computation chain from airfoil properties through to CL0.
    """
    
    def initialize(self):
        self.options.declare('d_alpha0_d_twist', default=-0.405, types=float)
        self.options.declare('N', default=1, desc='Number of nodes')

    def setup(self):

        N = self.options['N']

        self.add_subsystem('alpha0',
                        ZeroLiftAngle(d_alpha0_d_twist=self.options['d_alpha0_d_twist']),
                        promotes_inputs=['*'],
                        promotes_outputs=['*'])



        self.add_subsystem('cl0',
                        ZeroAngleLift(N=N),
                        promotes_inputs=['*'],
                        promotes_outputs=['*'])

# end

if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem(reports=False)
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    N = 1
    
    # Wing parameters
    ivc.add_output('alpha0_airfoil', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_twist', val=0.0, units='rad', desc='twist angle')

    ivc.add_output('CL_alpha_eff', val=5.0 * np.ones(N), units='1/rad', desc='Wing lift curve slope')
    
   

    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('CL0', GroupCL0(N=N), promotes=['*'])


    # Setup problem
    prob.setup()

    om.n2(prob.model)
    
    # Run baseline case
    prob.run_model()



    print('\nBaseline Configuration:')
    print('----------------------')
    print('Wing:')
    print(f'  Alpha_0:             {np.degrees(prob.get_val("w_alpha0.alpha0")[0]):8.3f} deg')
    print(f'  CL_0:                {prob.get_val("CL0_w")[0]:8.3f}')

    print('\nCanard:')
    print(f'  Alpha_0:             {np.degrees(prob.get_val("c_alpha0.alpha0")[0]):8.3f} deg')
    print(f'  CL_0:                {prob.get_val("CL0_c")[0]:8.3f}')

    print('\nTotal:')
    print(f'  CL_0:                {prob.get_val("CL0_total")[0]:8.3f}')
    
# Parameter sweeps

    # Wing zero-lift angle sweep
    alpha0_range = np.linspace(-4.0, 0.0, 50)
    CL0_w = []
    CL0_total = []
    for a0 in alpha0_range:
        prob.set_val('w_alpha0.alpha0_airfoil', a0)
        prob.run_model()
        CL0_w.append(prob.get_val('CL0_w')[0])
        CL0_total.append(prob.get_val('CL0_total')[0])

    plt.plot(alpha0_range, CL0_w, label='Wing')
    plt.plot(alpha0_range, CL0_total, '--', label='Total')
    plt.xlabel('Wing Zero-Lift Angle [deg]')
    plt.ylabel('$CL_{0}$')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Wing Zero-Lift Angle')

    plt.show()


# Mach number sweep

    prob.check_partials(compact_print=True)