import openmdao.api as om
import numpy as np

from .group_cl0 import GroupCL0
from .group_cl import GroupCL
from .group_cla import GroupClAlpha
from .lift_req import LiftRequired

class GroupLift(om.Group):
    """

    Group that computes required lift and lift coefficients for wing and canard.
    Computation chain:
    1. Required lift from weight and flight path
    2. Total lift coefficient required
    3. Zero-angle lift coefficients for wing and canard
    4. Lift curve slopes for wing and canard
    5. Individual lift coefficients for wing and canard
    """
    
    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
    
    def setup(self):
        # Step 1: Required lift force
        self.add_subsystem('lift_req',
                          LiftRequired(),
                          promotes_inputs=['mto', 'gamma'],
                          promotes_outputs=['lift'])
        
        # Step 2: Zero-angle lift coefficients
        self.add_subsystem('cl0',
                          GroupCL0(manta=self.options['manta'],
                                 ray=self.options['ray']),
                          promotes_inputs=['M_wing', 'tc_wing', 'alpha_L0_w',
                                         'M_canard', 'tc_canard', 'alpha_L0_c',
                                         'd_eps_w_d_alpha', 'd_eps_c_d_alpha',
                                         'S_c', 'S_w'],
                          promotes_outputs=['cl0_w', 'cl0_c', 'cl0_total'])
        
        # Step 3: Lift curve slopes
        self.add_subsystem('cl_alpha',
                          GroupClAlpha(manta=self.options['manta'],
                                     ray=self.options['ray']),
                          promotes_inputs=['CL_alpha_w', 'd_eps_w_d_alpha',
                                         'CL_alpha_c', 'd_eps_c_d_alpha',
                                         'S_c', 'S_w'],
                          promotes_outputs=['CL_alpha_w_eff', 'CL_alpha_c_eff',
                                         'CL_alpha_total'])
        
        # Step 4: Individual lift coefficients
        self.add_subsystem('cl',
                          GroupCL(),
                          promotes_inputs=['CL0_w', 'CL_alpha_w_eff',
                                         'CL0_c', 'CL_alpha_c_eff',
                                         'alpha'],
                          promotes_outputs=['CL_w', 'CL_c', 'CL_total'])


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Flight conditions
    ivc.add_output('mto', val=10000.0, units='kg', desc='Maximum takeoff weight')
    ivc.add_output('gamma', val=0.0, units='deg', desc='Flight path angle')
    ivc.add_output('alpha', val=2.0, units='deg', desc='Angle of attack')
    
    # Wing parameters
    ivc.add_output('M_wing', val=0.78, desc='Wing Mach number')
    ivc.add_output('tc_wing', val=0.12, desc='Wing thickness ratio')
    ivc.add_output('alpha_L0_w', val=-2.0, units='deg', desc='Wing zero-lift angle')
    ivc.add_output('CL_alpha_w', val=5.5, units='1/rad', desc='Wing lift curve slope')
    ivc.add_output('d_eps_w_d_alpha', val=0.25, desc='Wing downwash derivative')
    
    # Canard parameters
    ivc.add_output('M_canard', val=0.78, desc='Canard Mach number')
    ivc.add_output('tc_canard', val=0.10, desc='Canard thickness ratio')
    ivc.add_output('alpha_L0_c', val=-1.5, units='deg', desc='Canard zero-lift angle')
    ivc.add_output('CL_alpha_c', val=4.0, units='1/rad', desc='Canard lift curve slope')
    ivc.add_output('d_eps_c_d_alpha', val=0.1, desc='Canard downwash derivative')
    ivc.add_output('S_c', val=20.0, units='m**2', desc='Canard reference area')
    ivc.add_output('S_w', val=120.0, units='m**2', desc='Wing reference area')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('lift', GroupLift(manta=1), promotes=['*'])
    
    # Setup problem
    prob.setup()

    om.n2(prob)
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print('Flight Conditions:')
    print(f'  MTOW:                {prob.get_val("mto")[0]:8.1f} kg')
    print(f'  Flight Path Angle:   {prob.get_val("gamma")[0]:8.3f} deg')
    print(f'  Required Lift:       {prob.get_val("lift")[0]/1000:8.1f} kN')
    print(f'  Angle of Attack:     {prob.get_val("alpha")[0]:8.3f} deg')
    
    print('\nWing:')
    print(f'  CL0:                 {prob.get_val("cl0_w")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha_w_eff")[0]:8.3f} /rad')
    print(f'  CL:                  {prob.get_val("CL_w")[0]:8.3f}')
    
    print('\nCanard:')
    print(f'  CL0:                 {prob.get_val("cl0_c")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha_c_eff")[0]:8.3f} /rad')
    print(f'  CL:                  {prob.get_val("CL_c")[0]:8.3f}')
    
    print('\nTotal:')
    print(f'  CL0:                 {prob.get_val("cl0_total")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha_total")[0]:8.3f} /rad')
    print(f'  CL:                  {prob.get_val("CL_total")[0]:8.3f}')
    
    # Parameter sweeps
    import matplotlib.pyplot as plt
    
    # Create figure with 2x2 subplots
    plt.figure(figsize=(12, 10))
    
    # Alpha sweep
    alpha_range = np.linspace(-5.0, 10.0, 50)  # degrees
    CL_w = []
    CL_c = []
    CL_total = []
    for alpha in alpha_range:
        prob.set_val('alpha', alpha)
        prob.run_model()
        CL_w.append(prob.get_val('CL_w')[0])
        CL_c.append(prob.get_val('CL_c')[0])
        CL_total.append(prob.get_val('CL_total')[0])
    
    plt.subplot(221)
    plt.plot(alpha_range, CL_w, label='Wing')
    plt.plot(alpha_range, CL_c, label='Canard')
    plt.plot(alpha_range, CL_total, '--', label='Total')
    plt.xlabel('Angle of Attack [deg]')
    plt.ylabel('CL')
    plt.grid(True)
    plt.legend()
    plt.title('Lift Curves')
    
    # MTOW sweep
    prob.set_val('alpha', 2.0)  # Reset to baseline
    mtow_range = np.linspace(5000, 15000, 50)
    lift_req = []
    for mtow in mtow_range:
        prob.set_val('mto', mtow)
        prob.run_model()
        lift_req.append(prob.get_val('lift')[0])
    
    plt.subplot(222)
    plt.plot(mtow_range, np.array(lift_req)/1000)
    plt.xlabel('MTOW (kg)')
    plt.ylabel('Required Lift (kN)')
    plt.grid(True)
    plt.title('Effect of Maximum Takeoff Weight')
    
    # Area ratio sweep
    prob.set_val('mto', 10000)  # Reset to baseline
    Sc_range = np.linspace(10.0, 40.0, 50)
    CL_w = []
    CL_c = []
    CL_total = []
    for Sc in Sc_range:
        prob.set_val('S_c', Sc)
        prob.run_model()
        CL_w.append(prob.get_val('CL_w')[0])
        CL_c.append(prob.get_val('CL_c')[0])
        CL_total.append(prob.get_val('CL_total')[0])
    
    plt.subplot(223)
    plt.plot(Sc_range/prob.get_val('S_w')[0], CL_w, label='Wing')
    plt.plot(Sc_range/prob.get_val('S_w')[0], CL_c, label='Canard')
    plt.plot(Sc_range/prob.get_val('S_w')[0], CL_total, '--', label='Total')
    plt.xlabel('Area Ratio (Sc/Sw)')
    plt.ylabel('CL')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Canard Size')
    
    # Flight path angle sweep
    prob.set_val('S_c', 20.0)  # Reset to baseline
    gamma_range = np.linspace(-10, 10, 50)  # degrees
    lift_req = []
    for gamma in gamma_range: