import openmdao.api as om
import numpy as np
from .cl_alpha_airfoil import LiftCurveSlopeAirfoil
from .cl_alpha_eff import ComputeClAlphaWing, ComputeClAlphaCanard
from .zero_lift_ang import ZeroLiftAngle, ZeroAngleLift

class GroupCL(om.Group):

    """
    Group that computes total zero-angle lift coefficient by combining wing and canard.
    Includes computation chain from airfoil properties through to cl0.
    """
    
    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
    
    def setup(self):

        scaling_factors = [self.options['manta'], self.options['ray']]

        if self.options['manta'] == 1:
            self.add_subsystem('wing_cl_alpha_airfoil',
                          LiftCurveSlopeAirfoil(),
                          promotes_inputs=[('M', 'M_wing'),
                                         ('tc', 'tc_wing')],
                          promotes_outputs=[('cl_alpha', 'cl_alpha_airfoil_w')])

        
            self.add_subsystem('wing_cl_alpha',
                            ComputeClAlphaWing(ray=self.options['ray']),
                            promotes_inputs=['d_eps_w_d_alpha',
                                            ('CL_alpha_w', 'cl_alpha_airfoil_w')],
                            promotes_outputs=['CL_alpha_w_eff'])
            
            self.add_subsystem('wing_alpha0',
                            ZeroLiftAngle(),
                            promotes_inputs=[('alpha_L0_airfoil', 'alpha_L0_w')],
                            promotes_outputs=[('alpha0', 'alpha0_w')])
            
            self.add_subsystem('wing_cl0',
                            ZeroAngleLift(),
                            promotes_inputs=[('cl_alpha', 'CL_alpha_w_eff'),
                                            ('alpha0', 'alpha0_w')],
                            promotes_outputs=[('cl0', 'cl0_w')])
        # end
        
        if self.options['ray'] == 1:
            # Canard computation chain
            self.add_subsystem('canard_cl_alpha_airfoil',
                            LiftCurveSlopeAirfoil(),
                            promotes_inputs=[('M', 'M_canard'),
                                            ('tc', 'tc_canard')],
                            promotes_outputs=[('cl_alpha', 'cl_alpha_airfoil_c')])
            
            self.add_subsystem('canard_cl_alpha',
                            ComputeClAlphaCanard(manta=self.options['manta']),
                            promotes_inputs=['d_eps_c_d_alpha', 'S_c', 'S_w',
                                            ('CL_alpha_c', 'cl_alpha_airfoil_c')],
                            promotes_outputs=['CL_alpha_c_eff'])
            
            self.add_subsystem('canard_alpha0',
                            ZeroLiftAngle(),
                            promotes_inputs=[('alpha_L0_airfoil', 'alpha_L0_c')],
                            promotes_outputs=[('alpha0', 'alpha0_c')])
            
            self.add_subsystem('canard_cl0',
                            ZeroAngleLift(),
                            promotes_inputs=[('cl_alpha', 'CL_alpha_c_eff'),
                                            ('alpha0', 'alpha0_c')],
                            promotes_outputs=[('cl0', 'cl0_c')])
        # end
        
        # Sum the contributions
        adder = om.AddSubtractComp()
        adder.add_equation('cl0_total',
                          ['cl0_w', 'cl0_c'],
                          desc='Total zero-angle lift coefficient',
                          scaling_factors=scaling_factors)
        

        self.add_subsystem('sum_cl0', adder, promotes=['*'])


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Wing parameters
    ivc.add_output('M_wing', val=0.78, desc='Wing Mach number')
    ivc.add_output('tc_wing', val=0.12, desc='Wing thickness ratio')
    ivc.add_output('alpha_L0_w', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_eps_w_d_alpha', val=0.25, desc='Wing downwash derivative')
    
    # Canard parameters
    ivc.add_output('M_canard', val=0.78, desc='Canard Mach number')
    ivc.add_output('tc_canard', val=0.10, desc='Canard thickness ratio')
    ivc.add_output('alpha_L0_c', val=-1.5, units='deg', desc='Canard airfoil zero-lift angle')
    ivc.add_output('d_eps_c_d_alpha', val=0.1, desc='Canard downwash derivative')
    ivc.add_output('S_c', val=20.0, units='m**2', desc='Canard reference area')
    ivc.add_output('S_w', val=120.0, units='m**2', desc='Wing reference area')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cl', GroupCL(manta=1), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()

    
    print('\nBaseline Configuration:')
    print('----------------------')
    print('Wing:')
    print(f'  Airfoil cl_alpha:    {prob.get_val("cl_alpha_airfoil_w")[0]:8.3f} /rad')
    print(f'  Effective CL_alpha:   {prob.get_val("CL_alpha_w_eff")[0]:8.3f} /rad')
    print(f'  Alpha_0:             {np.degrees(prob.get_val("alpha0_w")[0]):8.3f} deg')
    print(f'  CL_0:                {prob.get_val("cl0_w")[0]:8.3f}')
    
    print('\nCanard:')
    print(f'  Airfoil cl_alpha:    {prob.get_val("cl_alpha_airfoil_c")[0]:8.3f} /rad')
    print(f'  Effective CL_alpha:   {prob.get_val("CL_alpha_c_eff")[0]:8.3f} /rad')
    print(f'  Alpha_0:             {np.degrees(prob.get_val("alpha0_c")[0]):8.3f} deg')
    print(f'  CL_0:                {prob.get_val("cl0_c")[0]:8.3f}')
    
    print('\nTotal:')
    print(f'  CL_0:                {prob.get_val("cl0_total")[0]:8.3f}')
    
    # Parameter sweeps
    import matplotlib.pyplot as plt
    
    # Create figure with 2x2 subplots
    plt.figure(figsize=(12, 10))
    
    # Wing thickness ratio sweep
    tc_range = np.linspace(0.08, 0.16, 50)
    cl0_w = []
    cl0_total = []
    for tc in tc_range:
        prob.set_val('tc_wing', tc)
        prob.run_model()
        cl0_w.append(prob.get_val('cl0_w')[0])
        cl0_total.append(prob.get_val('cl0_total')[0])
    
    plt.subplot(221)
    plt.plot(tc_range, cl0_w, label='Wing')
    plt.plot(tc_range, cl0_total, '--', label='Total')
    plt.xlabel('Wing Thickness Ratio')
    plt.ylabel('CL_0')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Wing Thickness')
    
    # Reset wing thickness
    prob.set_val('tc_wing', 0.12)
    
    # Wing zero-lift angle sweep
    alpha0_range = np.linspace(-4.0, 0.0, 50)
    cl0_w = []
    cl0_total = []
    for a0 in alpha0_range:
        prob.set_val('alpha_L0_w', a0)
        prob.run_model()
        cl0_w.append(prob.get_val('cl0_w')[0])
        cl0_total.append(prob.get_val('cl0_total')[0])
    
    plt.subplot(222)
    plt.plot(alpha0_range, cl0_w, label='Wing')
    plt.plot(alpha0_range, cl0_total, '--', label='Total')
    plt.xlabel('Wing Zero-Lift Angle [deg]')
    plt.ylabel('CL_0')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Wing Zero-Lift Angle')
    
    # Area ratio sweep
    Sc_range = np.linspace(10.0, 40.0, 50)
    cl0_c = []
    cl0_total = []
    for Sc in Sc_range:
        prob.set_val('S_c', Sc)
        prob.run_model()
        cl0_c.append(prob.get_val('cl0_c')[0])
        cl0_total.append(prob.get_val('cl0_total')[0])
    
    plt.subplot(223)
    plt.plot(Sc_range/prob.get_val('S_w')[0], cl0_c, label='Canard')
    plt.plot(Sc_range/prob.get_val('S_w')[0], cl0_total, '--', label='Total')
    plt.xlabel('Area Ratio (Sc/Sw)')
    plt.ylabel('CL_0')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Canard Size')
    
    # Mach number sweep
    M_range = np.linspace(0.3, 0.85, 50)
    cl0_w = []
    cl0_c = []
    cl0_total = []
    for M in M_range:
        prob.set_val('M_wing', M)
        prob.set_val('M_canard', M)
        prob.run_model()
        cl0_w.append(prob.get_val('cl0_w')[0])
        cl0_c.append(prob.get_val('cl0_c')[0])
        cl0_total.append(prob.get_val('cl0_total')[0])
    
    plt.subplot(224)
    plt.plot(M_range, cl0_w, label='Wing')
    plt.plot(M_range, cl0_c, label='Canard')
    plt.plot(M_range, cl0_total, '--', label='Total')
    plt.xlabel('Mach Number')
    plt.ylabel('CL_0')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Mach Number')
    
    plt.tight_layout()
    plt.show()