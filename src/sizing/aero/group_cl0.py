import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('../'))))

from cl_alpha_airfoil import LiftCurveSlope3D
from group_cla import CoupledCLAlphaWing, CoupledCLAlphaCanard, GroupCLAlpha
from zero_ang_lift import ZeroLiftAngle, ZeroAngleLift





class GroupCL0(om.Group):

    """
    Group that computes total zero-angle lift coefficient by combining wing and canard.
    Includes computation chain from airfoil properties through to CL0.
    """
    
    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
        self.options.declare('d_alpha0_d_twist', default=-0.405, types=float)
    def setup(self):

        scaling_factors = [self.options['manta'], self.options['ray']]

        

        if self.options['manta'] == 1:
            
            self.add_subsystem('w_alpha0',
                            ZeroLiftAngle(d_alpha0_d_twist=self.options['d_alpha0_d_twist']),
                            promotes_inputs=[],
                            promotes_outputs=[])

            self.add_subsystem('w_CL0',
                            ZeroAngleLift(),
                            promotes_inputs=[],
                            promotes_outputs=[])
            
            self.connect('w_alpha0.alpha0', 'w_CL0.alpha0')
        # end
        
        if self.options['ray'] == 1:
            # Canard computation chain
            self.add_subsystem('c_alpha0',
                            ZeroLiftAngle(),
                            promotes_inputs=[],
                            promotes_outputs=[])

            self.add_subsystem('c_CL0',
                            ZeroAngleLift(),
                            promotes_inputs=[],
                            promotes_outputs=[])
            
            self.connect('c_alpha0.alpha0', 'c_CL0.alpha0')
        # end
        
        if self.options['manta'] == 1 and self.options['ray'] == 1:
            self.connect('c_CL0.CL0', 'CL0_c')
            self.connect('w_CL0.CL0', 'CL0_w')
        # end

            # Sum the contributions
            adder = om.AddSubtractComp()
            adder.add_equation('CL0_total',
                            ['CL0_w', 'CL0_c'],
                            desc='Total zero-angle lift coefficient',
                            scaling_factors=scaling_factors)

            
            self.add_subsystem('sum_CL0', adder, promotes=['*'])

        # end


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem(reports=False)
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Wing parameters
    ivc.add_output('alpha0_airfoil_w', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_twist_w', val=0.0, units='rad', desc='twist angle')

    ivc.add_output('CL_alpha_w_eff', val=5.0, units='1/rad', desc='Wing lift curve slope')
    ivc.add_output('CL_alpha_c_eff', val=4.0, units='1/rad', desc='Canard lift curve slope')


    ivc.add_output('alpha0_airfoil_c', val=-1.5, units='deg', desc='Canard airfoil zero-lift angle')
    ivc.add_output('d_twist_c', val=0.0, units='rad', desc='twist angle')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])

    manta_opt = 1
    ray_opt = 1

    prob.model.add_subsystem('CL0', GroupCL0(manta=manta_opt,ray=ray_opt), promotes=['*'])


    if manta_opt == 1:
        prob.model.connect('alpha0_airfoil_w', 'w_alpha0.alpha0_airfoil')
        prob.model.connect('d_twist_w', 'w_alpha0.d_twist')
        prob.model.connect('CL_alpha_w_eff', 'w_CL0.CL_alpha')
    if ray_opt == 1:
        prob.model.connect('alpha0_airfoil_c', 'c_alpha0.alpha0_airfoil')
        prob.model.connect('CL_alpha_c_eff', 'c_CL0.CL_alpha')
        prob.model.connect('d_twist_c', 'c_alpha0.d_twist')
    # end



    # Setup problem
    prob.setup()

    om.n2(prob.model)
    
    # Run baseline case
    prob.run_model()



    if manta_opt == 1 and ray_opt == 1:
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