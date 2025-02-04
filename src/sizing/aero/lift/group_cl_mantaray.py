import openmdao.api as om
import numpy as np




import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))


from src.sizing.mission.atmos import ComputeAtmos
from src.sizing.mission.mach_number import MachNumber
from src.sizing.aero.lift.lift_req import LiftRequired
from src.sizing.aero.lift.group_cla_mantaray import GroupCLAlphaMantaRay 
from src.sizing.aero.lift.group_cl0_mantaray import GroupCL0MantaRay
from src.sizing.aero.lift.ang_attack import AngleOfAttack

from src.sizing.aero.lift.cl import LiftCoefficient



class GroupCLMantaRay(om.Group):

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
        self.options.declare('N', default=1, desc='Number of nodes')

    def setup(self):

        N = self.options['N']

        self.add_subsystem('lift_req', LiftRequired(N=N),
                          promotes_inputs=['gamma'],
                          promotes_outputs=['*'])

                # Step 3: Lift curve slopes
        self.add_subsystem('cl_alpha',
                          GroupCLAlphaMantaRay(manta=self.options['manta'],ray=self.options['ray'], N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

        

        self.add_subsystem('cl0',
                          GroupCL0MantaRay(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
        


        self.add_subsystem('alpha', AngleOfAttack(N=N), promotes_inputs=['*'],
                     promotes_outputs=['*'])
        


        # Step 4: Individual lift coefficients
        self.add_subsystem('cl_mantaray',
                          LiftCoefficient(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

        # Step 4: Manta lift coefficient
        self.add_subsystem('cl_manta',
                          LiftCoefficient(N=N),
                          promotes_inputs=[],
                          promotes_outputs=[])
        
        # Step 4: Ray lift coefficient
        self.add_subsystem('cl_ray',
                          LiftCoefficient(N=N),
                          promotes_inputs=[],
                          promotes_outputs=[])
        

        self.connect('CL0_total', 'CL0')
        self.connect('CL_alpha_total', ['CL_alpha','CL_alpha_eff'])
        self.connect('cl_alpha_manta.CL_alpha_eff', 'manta_cl0.CL_alpha_eff')
        self.connect('cl_alpha_ray.CL_alpha_eff', 'ray_cl0.CL_alpha_eff')

        self.connect('cl_alpha_manta.CL_alpha_eff', 'cl_manta.CL_alpha_eff')
        self.connect('cl_alpha_ray.CL_alpha_eff', 'cl_ray.CL_alpha_eff')

        self.connect('manta_cl0.CL0', 'cl_manta.CL0')
        self.connect('ray_cl0.CL0', 'cl_ray.CL0')

        self.connect('alpha', ['cl_manta.alpha', 'cl_ray.alpha'])






        # end
    # end


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()

    N = 1

    # Flight conditions
    ivc.add_output('alt', val=30000 * np.ones(N), units='ft', desc='Altitude')
    ivc.add_output('u', val=50.0 * np.ones(N), units='m/s', desc='Flow speed')
    ivc.add_output('mach', val=0.8 * np.ones(N), units='m', desc='Mach number')
    ivc.add_output('rho', val=0.8 * np.ones(N), units='kg/m**3', desc='Density')



    ivc.add_output('mu', val=1.789e-5 * np.ones(N), units='Pa*s', desc='Dynamic viscosity')

    ivc.add_output('gamma', val=0.0 * np.ones(N), units='deg', desc='Flight path angle')

    # Manta parameters
    ivc.add_output('mto', val=10000.0, units='kg', desc='Maximum takeoff weight')

    ivc.add_output('S_ref', val=100.0, units='m**2', desc='Reference area')
    ivc.add_output('phi_50_manta', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil_manta', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')

    ivc.add_output('aspect_ratio_manta', val=10.0, desc='Manta aspect ratio')
    ivc.add_output('d_eps_manta_d_alpha', val=0.25, desc='Manta downwash derivative')
    ivc.add_output('alpha0_airfoil_manta', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')

    ivc.add_output('d_twist_manta', val=0.0, units='rad', desc='twist angle')


    # Ray Parameters
    ivc.add_output('S_ray', val=100.0, units='m**2', desc='Ray area')
    ivc.add_output('phi_50_ray', val=5.0, units='deg', desc='Ray 50% chord sweep angle')

    ivc.add_output('cl_alpha_airfoil_ray', val=2*np.pi, units='1/rad', desc='Ray airfoil lift curve slope')
    ivc.add_output('aspect_ratio_ray', val=10.0, desc='Ray aspect ratio')
    ivc.add_output('d_eps_ray_d_alpha', val=0.25, desc='Ray downwash derivative')

    ivc.add_output('alpha0_airfoil_ray', val=-2.0, units='deg', desc='Ray airfoil zero-lift angle')
    ivc.add_output('d_twist_ray', val=0.0, units='rad', desc='twist angle')



    manta_opt = 1
    ray_opt = 1
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cl', GroupCLMantaRay(manta=manta_opt, ray = ray_opt, N=N), promotes=['*'])


    prob.model.connect('mach', ['cl_alpha_manta.mach', 'cl_alpha_ray.mach'])


    prob.model.connect('phi_50_manta', 'cl_alpha_manta.phi_50')
    prob.model.connect('phi_50_ray', 'cl_alpha_ray.phi_50')

    prob.model.connect('cl_alpha_airfoil_manta', 'cl_alpha_manta.cl_alpha_airfoil')
    prob.model.connect('cl_alpha_airfoil_ray', 'cl_alpha_ray.cl_alpha_airfoil')

    prob.model.connect('alpha0_airfoil_manta', 'manta_cl0.alpha0_airfoil')
    prob.model.connect('alpha0_airfoil_ray', 'ray_cl0.alpha0_airfoil')

    prob.model.connect('d_twist_manta', 'manta_cl0.d_twist')
    prob.model.connect('d_twist_ray', 'ray_cl0.d_twist')

    prob.model.connect('aspect_ratio_manta', 'cl_alpha_manta.aspect_ratio')  
    prob.model.connect('aspect_ratio_ray', 'cl_alpha_ray.aspect_ratio')

    prob.model.connect('S_ref', 'S_manta')




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
    print(f'  CL0:                 {prob.get_val("manta_cl0.CL0")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("cl_alpha_manta.CL_alpha_eff")[0]:8.3f} /rad')

    


    print('\nCanard:')
    print(f'  CL0:                 {prob.get_val("ray_cl0.CL0")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("cl_alpha_ray.CL_alpha_eff")[0]:8.3f} /rad')




    print('\nTotal:')
    print(f'  CL0:                 {prob.get_val("CL0_total")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha_total")[0]:8.3f} /rad')
    print(f'  CL:                  {prob.get_val("CL")[0]:8.3f}')
    
    prob.check_partials(compact_print=True)

    