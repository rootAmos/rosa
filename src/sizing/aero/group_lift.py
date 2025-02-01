import openmdao.api as om
import numpy as np

from group_cl0 import GroupCL0
from group_cl import GroupCL
from group_cla import GroupCLAlpha
from lift_req import LiftRequired
from ang_attack import AngleOfAttack


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
    

        if self.options['manta'] == 1 and self.options['ray'] == 0:

            cl0_outputs =  ['CL0_w']
            clalpha_outputs = ['CL_alpha_w_eff']
            cl_outputs = ['CL_w']

            self.connect('CL0_w', 'alpha.CL0')
            self.connect('CL_alpha_w_eff', 'alpha.CL_alpha')

            self.connect('CL0_w', 'cl.CL0_w')
            self.connect('cl_alpha.CL_alpha_w_eff', 'cl.CL_alpha_eff')


        elif self.options['ray'] == 1 and self.options['manta'] == 0:

            cl0_outputs = ['CL0_c']
            clalpha_outputs = ['CL_alpha_c_eff']
            cl_outputs = ['CL_c']
            cl_inputs = ['CL_c']

            self.connect('CL_alpha_c_eff', ['alpha.CL_alpha','cl.CL_alpha_eff'])

            self.connect('c_CL0.CL0', ['cl.CL0_c','alpha.CL0'])



        elif self.options['manta'] == 1 and self.options['ray'] == 1:

            cl0_outputs = ['CL0_total']
            cl0_inputs = ['CL0_w', 'CL0_c']
            clalpha_inputs = []
            clalpha_outputs = ['CL_alpha_total']
            cl_outputs = ['CL_total']
            cl_inputs = ['CL_w', 'CL_c','alpha']

            self.connect('cl0.c_CL0.CL0', 'cl.canard_cl.CL0')
            self.connect('cl0.w_CL0.CL0', 'cl.wing_cl.CL0')

            self.connect('cl_alpha.CL_alpha_w_eff', ['cl0.w_CL0.CL_alpha','cl.wing_cl.CL_alpha_eff'])
            self.connect('cl_alpha.CL_alpha_c_eff', ['cl0.c_CL0.CL_alpha','cl.canard_cl.CL_alpha_eff'])

            self.connect('CL0_total', 'alpha.CL0')
            self.connect('CL_alpha_total', ['alpha.CL_alpha'])
        # end


        # Step 3: Lift curve slopes
        self.add_subsystem('cl_alpha',
                          GroupCLAlpha(manta=self.options['manta'],
                                     ray=self.options['ray']),
                          promotes_inputs=clalpha_inputs,
                          promotes_outputs=clalpha_outputs)
        
        
        # Step 2: Zero-angle lift coefficients
        self.add_subsystem('cl0',
                          GroupCL0(manta=self.options['manta'],
                                 ray=self.options['ray']),
                          promotes_inputs=cl0_inputs,
                          promotes_outputs=cl0_outputs)
        
        self.add_subsystem('alpha', AngleOfAttack(), promotes_inputs=['rho', 'u', 'lift'],
                     promotes_outputs=['alpha'])
        

        # Step 4: Individual lift coefficients
        self.add_subsystem('cl',
                          GroupCL(manta=self.options['manta'],
                                 ray=self.options['ray']),
                          promotes_inputs=cl_inputs,
                          promotes_outputs=cl_outputs)
        


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Flight conditions
    ivc.add_output('mto_m', val=10000.0, units='kg', desc='Maximum takeoff weight of Manta')
    ivc.add_output('mto_r', val=5000.0, units='kg', desc='Maximum takeoff weight of Ray')
    ivc.add_output('mto_mr', val=10000.0, units='kg', desc='Maximum takeoff weight of Manta Ray')

    ivc.add_output('gamma', val=0.0, units='deg', desc='Flight path angle')
    ivc.add_output('rho', val=0.38, units='kg/m**3', desc='Air density')
    ivc.add_output('u', val=70.0, units='m/s', desc='Flight speed')



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

    manta_opt = 1
    ray_opt = 1
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('lift', GroupLift(manta=manta_opt, ray = ray_opt), promotes=['*'])



    prob.model.connect('mach_w', 'cl_alpha.w_cl_alpha_3d.mach')
    prob.model.connect('phi_50_w', 'cl_alpha.w_cl_alpha_3d.phi_50')
    prob.model.connect('aspect_ratio_w', 'cl_alpha.w_cl_alpha_3d.aspect_ratio')

    prob.model.connect('cl_alpha_airfoil_w', 'cl_alpha.w_cl_alpha_3d.cl_alpha_airfoil')

    prob.model.connect('alpha0_airfoil_w', 'cl0.w_alpha0.alpha0_airfoil')
    prob.model.connect('d_twist_w', 'cl0.w_alpha0.d_twist')

    prob.model.connect('mach_c', 'cl_alpha.c_cl_alpha_3d.mach')
    prob.model.connect('phi_50_c', 'cl_alpha.c_cl_alpha_3d.phi_50')

    prob.model.connect('cl_alpha_airfoil_c', 'cl_alpha.c_cl_alpha_3d.cl_alpha_airfoil')
    prob.model.connect('aspect_ratio_c', 'cl_alpha.c_cl_alpha_3d.aspect_ratio')
    prob.model.connect('alpha0_airfoil_c', 'cl0.c_alpha0.alpha0_airfoil')

    prob.model.connect('d_eps_w_d_alpha', 'cl_alpha.d_eps_w_d_alpha')
    prob.model.connect('d_eps_c_d_alpha', 'cl_alpha.d_eps_c_d_alpha')
    prob.model.connect('d_twist_c', 'cl0.c_alpha0.d_twist')



    if manta_opt == 1 and ray_opt == 0:
        prob.model.connect('mto_m', 'mto')
        prob.model.connect('S_w', ['alpha.S_ref','cl_alpha.S_w'])
    elif ray_opt == 1 and manta_opt == 0:
        prob.model.connect('mto_r', 'mto')
        prob.model.connect('S_c', ['alpha.S_ref','cl_alpha.S_c'])
    elif manta_opt == 1 and ray_opt == 1:
        prob.model.connect('S_w', ['alpha.S_ref','cl_alpha.S_w'])
        prob.model.connect('S_c', ['cl_alpha.S_c'])
        prob.model.connect('mto_mr', 'mto')
    # end



    # Setup problem
    prob.setup()


    #om.n2(prob)
    
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
    print(f'  CL0:                 {prob.get_val("cl0.w_CL0.CL0")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("cl_alpha.CL_alpha_w_eff")[0]:8.3f} /rad')
    print(f'  CL:                  {prob.get_val("cl.CL_w")[0]:8.3f}')
    


    print('\nCanard:')
    print(f'  CL0:                 {prob.get_val("cl0.c_CL0.CL0")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("cl_alpha.CL_alpha_c_eff")[0]:8.3f} /rad')
    print(f'  CL:                  {prob.get_val("cl.CL_c")[0]:8.3f}')
    


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

    prob.check_partials(compact_print=True)
    plt.show()
    
    