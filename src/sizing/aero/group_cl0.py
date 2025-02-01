import openmdao.api as om
import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('../'))))

from cl_alpha_airfoil import LiftCurveSlope3D
from group_cla import CoupledCLAlphaWing, CoupledCLAlphaCanard
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
            self.add_subsystem('w_cl_alpha_3d',
                          LiftCurveSlope3D(),
                          promotes_inputs=[],
                          promotes_outputs=[])


        
            self.add_subsystem('w_cl_alpha_cpld',
                            CoupledCLAlphaWing(ray=self.options['ray']),
                            promotes_inputs=['*'],
                            promotes_outputs=['*'])

            


            self.add_subsystem('w_alpha0',
                            ZeroLiftAngle(d_alpha0_d_twist=self.options['d_alpha0_d_twist']),
                            promotes_inputs=[],
                            promotes_outputs=[])

            

            self.add_subsystem('w_CL0',
                            ZeroAngleLift(),
                            promotes_inputs=[],
                            promotes_outputs=[])
            
            self.connect('w_alpha0.alpha0', 'w_CL0.alpha0')
            self.connect('CL_alpha_w_eff', 'w_CL0.CL_alpha')
            self.connect('w_cl_alpha_3d.CL_alpha', 'CL_alpha_w')
            self.connect('w_CL0.CL0', 'CL0_w')
        # end
        
        if self.options['ray'] == 1:
            # Canard computation chain
            self.add_subsystem('c_cl_alpha_3d',
                            LiftCurveSlope3D(),
                            promotes_inputs=[],
                            promotes_outputs=[])

            self.add_subsystem('c_cl_alpha_cpld',
                            CoupledCLAlphaCanard(manta=self.options['manta']),
                            promotes_inputs=['*'],
                            promotes_outputs=['*'])

            self.add_subsystem('c_alpha0',
                            ZeroLiftAngle(),
                            promotes_inputs=[],
                            promotes_outputs=[])

            self.add_subsystem('c_CL0',
                            ZeroAngleLift(),
                            promotes_inputs=[],
                            promotes_outputs=[])
            
            self.connect('c_alpha0.alpha0', 'c_CL0.alpha0')
            self.connect('CL_alpha_c_eff', 'c_CL0.CL_alpha')
            self.connect('c_cl_alpha_3d.CL_alpha', 'CL_alpha_c')
            self.connect('c_CL0.CL0', 'CL0_c')


        # end
        

        # Sum the contributions

        adder = om.AddSubtractComp()
        adder.add_equation('CL0_total',
                          ['CL0_w', 'CL0_c'],
                          desc='Total zero-angle lift coefficient',
                          scaling_factors=scaling_factors)


        

        self.add_subsystem('sum_CL0', adder, promotes=['*'])


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem(reports=False)
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Wing parameters
    ivc.add_output('mach_w', val=0.78, desc='Wing Mach number')
    ivc.add_output('phi_50_w', val=5.0, units='deg', desc='Wing 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil_w', val=2*np.pi, units='1/rad', desc='Wing airfoil lift curve slope')
    ivc.add_output('alpha0_airfoil_w', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_twist_w', val=0.0, units='rad', desc='twist angle')
    ivc.add_output('aspect_ratio_w', val=10.0, desc='Wing aspect ratio')
    ivc.add_output('d_eps_w_d_alpha', val=0.25, desc='Wing downwash derivative')

    # Canard parameters
    ivc.add_output('mach_c', val=0.78, desc='Canard Mach number')
    ivc.add_output('phi_50_c', val=4.0, units='deg', desc='Canard 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil_c', val=2*np.pi, units='1/rad', desc='Canard airfoil lift curve slope')
    ivc.add_output('alpha0_airfoil_c', val=-1.5, units='deg', desc='Canard airfoil zero-lift angle')
    ivc.add_output('d_twist_c', val=0.0, units='rad', desc='twist angle')
    ivc.add_output('aspect_ratio_c', val=10.0, desc='Canard aspect ratio')
    ivc.add_output('d_eps_c_d_alpha', val=0.1, desc='Canard downwash derivative')
    
    ivc.add_output('S_c', val=20.0, units='m**2', desc='Canard reference area')
    ivc.add_output('S_w', val=120.0, units='m**2', desc='Wing reference area')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('CL0', GroupCL0(manta=1,ray=1), promotes=['*'])


    prob.model.connect('mach_w', 'w_cl_alpha_3d.mach')
    prob.model.connect('phi_50_w', 'w_cl_alpha_3d.phi_50')
    prob.model.connect('cl_alpha_airfoil_w', 'w_cl_alpha_3d.cl_alpha_airfoil')
    prob.model.connect('aspect_ratio_w', 'w_cl_alpha_3d.aspect_ratio')
    prob.model.connect('alpha0_airfoil_w', 'w_alpha0.alpha0_airfoil')
    prob.model.connect('d_twist_w', 'w_alpha0.d_twist')

    prob.model.connect('mach_c', 'c_cl_alpha_3d.mach')
    prob.model.connect('phi_50_c', 'c_cl_alpha_3d.phi_50')
    prob.model.connect('cl_alpha_airfoil_c', 'c_cl_alpha_3d.cl_alpha_airfoil')

    prob.model.connect('aspect_ratio_c', 'c_cl_alpha_3d.aspect_ratio')
    prob.model.connect('alpha0_airfoil_c', 'c_alpha0.alpha0_airfoil')
    prob.model.connect('d_twist_c', 'c_alpha0.d_twist')

    # Setup problem
    prob.setup()

    #om.n2(prob.model)
    
    # Run baseline case
    prob.run_model()



    
    print('\nBaseline Configuration:')
    print('----------------------')
    print('Wing:')
    print(f'  Airfoil cl_alpha:    {prob.get_val("cl_alpha_airfoil_w")[0]:8.3f} /rad')
    print(f'  Effective CL_alpha:   {prob.get_val("CL_alpha_w_eff")[0]:8.3f} /rad')
    print(f'  Alpha_0:             {np.degrees(prob.get_val("w_alpha0.alpha0")[0]):8.3f} deg')
    print(f'  CL_0:                {prob.get_val("CL0_w")[0]:8.3f}')
    
    print('\nCanard:')
    print(f'  Airfoil cl_alpha:    {prob.get_val("cl_alpha_airfoil_c")[0]:8.3f} /rad')
    print(f'  Effective CL_alpha:   {prob.get_val("CL_alpha_c_eff")[0]:8.3f} /rad')
    print(f'  Alpha_0:             {np.degrees(prob.get_val("c_alpha0.alpha0")[0]):8.3f} deg')
    print(f'  CL_0:                {prob.get_val("CL0_c")[0]:8.3f}')
    

    print('\nTotal:')
    print(f'  CL_0:                {prob.get_val("CL0_total")[0]:8.3f}')
    
    # Parameter sweeps
    import matplotlib.pyplot as plt
    
    # Wing zero-lift angle sweep
    alpha0_range = np.linspace(-4.0, 0.0, 50)
    CL0_w = []
    CL0_total = []
    for a0 in alpha0_range:
        prob.set_val('w_alpha0.alpha0_airfoil', a0)
        prob.run_model()
        CL0_w.append(prob.get_val('CL0_w')[0])
        CL0_total.append(prob.get_val('CL0_total')[0])


    
    plt.subplot(222)
    plt.plot(alpha0_range, CL0_w, label='Wing')
    plt.plot(alpha0_range, CL0_total, '--', label='Total')
    plt.xlabel('Wing Zero-Lift Angle [deg]')
    plt.ylabel('CL_0')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Wing Zero-Lift Angle')
    
    # Area ratio sweep
    Sc_range = np.linspace(10.0, 40.0, 50)
    CL0_c = []
    CL0_total = []
    for Sc in Sc_range:
        prob.set_val('S_c', Sc)
        prob.run_model()
        CL0_c.append(prob.get_val('CL0_c')[0])
        CL0_total.append(prob.get_val('CL0_total')[0])
    
    plt.subplot(223)
    plt.plot(Sc_range/prob.get_val('S_w')[0], CL0_c, label='Canard')
    plt.plot(Sc_range/prob.get_val('S_w')[0], CL0_total, '--', label='Total')
    plt.xlabel('Area Ratio (Sc/Sw)')
    plt.ylabel('CL_0')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Canard Size')
    
    # Mach number sweep
    M_range = np.linspace(0.3, 0.85, 50)
    CL0_w = []
    CL0_c = []
    CL0_total = []
    for M in M_range:
        prob.set_val('w_cl_alpha_3d.mach', M)
        prob.set_val('c_cl_alpha_3d.mach', M)
        prob.run_model()
        CL0_w.append(prob.get_val('CL0_w')[0])
        CL0_c.append(prob.get_val('CL0_c')[0])
        CL0_total.append(prob.get_val('CL0_total')[0])
    
    plt.subplot(224)
    plt.plot(M_range, CL0_w, label='Wing')
    plt.plot(M_range, CL0_c, label='Canard')
    plt.plot(M_range, CL0_total, '--', label='Total')
    plt.xlabel('Mach Number')
    plt.ylabel('CL_0')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Mach Number')
    
    plt.tight_layout()
    plt.show()

    prob.check_partials(compact_print=True)