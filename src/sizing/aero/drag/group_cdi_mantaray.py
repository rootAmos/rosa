import openmdao.api as om


import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.sizing.aero.drag.group_cdi_manta import GroupCDiManta
from src.sizing.aero.drag.group_cdi_ray import GroupCDiRay


import numpy as np



class GroupCDiMantaRay(om.Group):

    """

    Group that computes induced drag for a lifting surface.
    Includes zero-lift angle calculation, total lift calculation, 
    Oswald efficiency, and induced drag computation.
    """
    
    def initialize(self):
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
        self.options.declare('N', default=1, desc='Number of nodes')

    def setup(self):

        self.add_subsystem('manta', GroupCDiManta(ray=self.options['ray'], N=self.options['N']),
                           promotes=[])
        
        self.add_subsystem('ray', GroupCDiRay(manta=self.options['manta'], N=self.options['N']),
                           promotes=[])



        adder = om.AddSubtractComp()
        adder.add_equation('CDi_total',
                        ['CDi_manta', 'CDi_ray'],
                        desc='Total induced drag coefficient',
                        vec_size=self.options['N'])

        self.add_subsystem('sum_cdi', adder, promotes=['*'])

        self.connect('manta.CDi', 'CDi_manta')
        self.connect('ray.CDi', 'CDi_ray')


if __name__ == "__main__":

    # Create problem instance
    prob = om.Problem(reports=False)
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()

    N = 1
    
    # Wing parameters
    #ivc.add_output('alpha0_airfoil_manta', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    #ivc.add_output('alpha0_airfoil_ray', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')

    #ivc.add_output('d_twist_manta', val=0.0, units='rad', desc='twist angle')
    #ivc.add_output('d_twist_ray', val=0.0, units='rad', desc='twist angle')
    #ivc.add_output('CL_alpha_eff', val=5.0, units='1/rad', desc='Wing lift curve slope')
   


    #ivc.add_output('mach', val=0.78 * np.ones(N), desc='Manta Mach number')
    #ivc.add_output('phi_50_manta', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    #ivc.add_output('phi_50_ray', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    #ivc.add_output('cl_alpha_airfoil_manta', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')

    #ivc.add_output('cl_alpha_airfoil_ray', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')


    #ivc.add_output('d_eps_manta_d_alpha', val=0.25, desc='Manta downwash derivative')
    #ivc.add_output('d_eps_ray_d_alpha', val=0.25, desc='Manta downwash derivative')

    ivc.add_output('aspect_ratio_manta', val=5.0, desc='aspect_ratiospect ratio')
    ivc.add_output('aspect_ratio_ray', val=5.0, desc='aspect_ratiospect ratio')
    ivc.add_output('lambda_manta', val=0.45, desc='Taper ratio of manta')
    ivc.add_output('lambda_ray', val=0.45, desc='Taper ratio of ray')

    ivc.add_output('sweep_25_manta', val=12.0, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('sweep_25_ray', val=12.0, units='deg', desc='Quarter-chord sweep angle')



    ivc.add_output('h_winglet_manta', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('h_winglet_ray', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('span_manta', val=30.0, units='m', desc='Wing span')
    ivc.add_output('span_ray', val=30.0, units='m', desc='Wing span')
    ivc.add_output('k_WL_manta', val=2.83, desc='Winglet effectiveness factor')
    ivc.add_output('k_WL_ray', val=2.83, desc='Winglet effectiveness factor')


    ##ivc.add_output('lift_manta', val=10000 * np.ones(N), units='N', desc='Lift')
    #ivc.add_output('lift_ray', val=10000 * np.ones(N), units='N', desc='Lift')
    #ivc.add_output('rho', val=0.3639 * np.ones(N), units='kg/m**3', desc='Density')
    #ivc.add_output('u', val=50.0 * np.ones(N), units='m/s', desc='Flow speed')


    #ivc.add_output('S_manta', val=100.0, units='m**2', desc='Manta reference area')
    #ivc.add_output('S_ray', val=100.0, units='m**2', desc='Ray reference area')

    ivc.add_output('CL_manta', val=0.2, desc='manta lift coefficient')
    ivc.add_output('CL0_manta', val=0.02, desc='manta zero lift coefficient')


    ivc.add_output('CL_ray', val=0.2, desc='ray lift coefficient')
    ivc.add_output('CL0_ray', val=0.02, desc='ray zero lift coefficient')



    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cdi_manta_ray', GroupCDiMantaRay(ray=1,manta=1, N=N), promotes=['*'])

    prob.model.connect('CL_manta', 'manta.CL')
    prob.model.connect('CL0_manta', 'manta.CL0')
    prob.model.connect('CL_ray', 'ray.CL')
    prob.model.connect('CL0_ray', 'ray.CL0')
    prob.model.connect('lambda_manta', 'manta.lambda')
    prob.model.connect('lambda_ray', 'ray.lambda')
    #prob.model.connect('S_ray', 'ray.S_ray')
    #prob.model.connect('S_manta', ['S_ref','ray.S_manta'])

    #prob.model.connect('alpha0_airfoil_manta', 'manta.alpha0_airfoil')
    #prob.model.connect('alpha0_airfoil_ray', 'ray.alpha0_airfoil')

    #prob.model.connect('phi_50_manta', 'manta.phi_50')
    #prob.model.connect('phi_50_ray', 'ray.phi_50')

    #prob.model.connect('d_twist_manta', 'manta.d_twist')
    #prob.model.connect('d_twist_ray', 'ray.d_twist')

    #prob.model.connect('cl_alpha_airfoil_manta', 'manta.cl_alpha_airfoil')
    #prob.model.connect('cl_alpha_airfoil_ray', 'ray.cl_alpha_airfoil')


    prob.model.connect('aspect_ratio_manta', 'manta.aspect_ratio')
    prob.model.connect('aspect_ratio_ray', 'ray.aspect_ratio')
    
    #prob.model.connect('taper_manta', 'manta.taper')
    #prob.model.connect('taper_ray', 'ray.taper')

    prob.model.connect('sweep_25_manta', 'manta.sweep_25')
    prob.model.connect('sweep_25_ray', 'ray.sweep_25')

    prob.model.connect('h_winglet_manta', 'manta.h_winglet')
    prob.model.connect('h_winglet_ray', 'ray.h_winglet')

    prob.model.connect('span_manta', 'manta.span')
    prob.model.connect('span_ray', 'ray.span')

    prob.model.connect('k_WL_manta', 'manta.k_WL')
    prob.model.connect('k_WL_ray', 'ray.k_WL')

    #prob.model.connect('lift_manta', 'manta.lift')
    #prob.model.connect('lift_ray', 'ray.lift')

    prob.setup()

    om.n2(prob.model)

    prob.run_model()

    prob.check_partials(compact_print=True)


    
    
    
    
    
    
    
    
    
    
    




    # Setup problem
    prob.setup()

    om.n2(prob.model)
    
    # Run baseline case
    prob.run_model()

    #prob.check_partials(compact_print=True)


