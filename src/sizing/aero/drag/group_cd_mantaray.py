import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt


import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))



from src.sizing.aero.drag.group_cdi_mantaray import GroupCDiMantaRay
from src.sizing.aero.drag.group_cd0_mantaray import GroupCD0MantaRay




class GroupCDMantaRay(om.Group):
    """
    Group that computes total induced drag coefficient by summing contributions
    from wing and canard.
    """
    def initialize(self):

        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')



    def setup(self):
        # Wing induced drag

        self.add_subsystem('cdi_manta_ray', GroupCDiMantaRay(manta=self.options['manta'], ray=self.options['ray']), promotes_inputs=['*'], promotes_outputs=['*'])   
        self.add_subsystem('cd0_manta_ray', GroupCD0MantaRay(), promotes_inputs=['*'], promotes_outputs=['*'])
        # end

        # Sum the contributions
        adder = om.AddSubtractComp()
        adder.add_equation('CD_manta_ray',
                        ['CD0_manta_ray', 'CDi_manta_ray'],
                        desc='Total drag coefficient')


        self.add_subsystem('sum', adder, promotes=['*'])
        # end

        self.connect('sum.CD0', 'CD0_manta_ray')
        self.connect('CDi_total', 'CDi_manta_ray')



if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
        # Wing parameters
    ivc.add_output('alpha0_airfoil_manta', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('alpha0_airfoil_ray', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_twist_manta', val=0.0, units='rad', desc='twist angle')
    ivc.add_output('d_twist_ray', val=0.0, units='rad', desc='twist angle')
    #ivc.add_output('CL_alpha_eff', val=5.0, units='1/rad', desc='Wing lift curve slope')
   


    ivc.add_output('phi_50_manta', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    ivc.add_output('phi_50_ray', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil_manta', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')
    ivc.add_output('cl_alpha_airfoil_ray', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')


    ivc.add_output('d_eps_manta_d_alpha', val=0.25, desc='Manta downwash derivative')
    ivc.add_output('d_eps_ray_d_alpha', val=0.25, desc='Manta downwash derivative')

    ivc.add_output('aspect_ratio_manta', val=5.0, desc='aspect_ratiospect ratio')
    ivc.add_output('aspect_ratio_ray', val=5.0, desc='aspect_ratiospect ratio')
    ivc.add_output('taper_manta', val=0.45, desc='Taper ratio')
    ivc.add_output('taper_ray', val=0.45, desc='Taper ratio')
    ivc.add_output('sweep_25_manta', val=12.0, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('sweep_25_ray', val=12.0, units='deg', desc='Quarter-chord sweep angle')



    ivc.add_output('h_winglet_manta', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('h_winglet_ray', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('span_manta', val=30.0, units='m', desc='Wing span')
    ivc.add_output('span_ray', val=30.0, units='m', desc='Wing span')
    ivc.add_output('k_WL_manta', val=2.83, desc='Winglet effectiveness factor')
    ivc.add_output('k_WL_ray', val=2.83, desc='Winglet effectiveness factor')


    ivc.add_output('lift_manta', val=10000, units='N', desc='Lift')
    ivc.add_output('lift_ray', val=10000, units='N', desc='Lift')

    ivc.add_output('S_manta', val=100.0, units='m**2', desc='Manta reference area')
    ivc.add_output('S_ray', val=100.0, units='m**2', desc='Ray reference area')

    
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


    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('CD', GroupCDMantaRay(manta=1, ray=1), promotes=['*'])
    
    

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

    
    prob.model.connect('S_ray', 'ray.S_ray')
    prob.model.connect('S_manta', ['S_ref','ray.S_manta'])


    prob.model.connect('alpha0_airfoil_manta', 'manta.alpha0_airfoil')
    prob.model.connect('alpha0_airfoil_ray', 'ray.alpha0_airfoil')

    prob.model.connect('phi_50_manta', 'manta.phi_50')
    prob.model.connect('phi_50_ray', 'ray.phi_50')

    prob.model.connect('d_twist_manta', 'manta.d_twist')
    prob.model.connect('d_twist_ray', 'ray.d_twist')

    prob.model.connect('cl_alpha_airfoil_manta', 'manta.cl_alpha_airfoil')
    prob.model.connect('cl_alpha_airfoil_ray', 'ray.cl_alpha_airfoil')


    prob.model.connect('aspect_ratio_manta', 'manta.aspect_ratio')
    prob.model.connect('aspect_ratio_ray', 'ray.aspect_ratio')
    
    prob.model.connect('taper_manta', 'manta.taper')
    prob.model.connect('taper_ray', 'ray.taper')

    prob.model.connect('sweep_25_manta', 'manta.sweep_25')
    prob.model.connect('sweep_25_ray', 'ray.sweep_25')

    prob.model.connect('h_winglet_manta', 'manta.h_winglet')
    prob.model.connect('h_winglet_ray', 'ray.h_winglet')

    prob.model.connect('span_manta', 'manta.span')
    prob.model.connect('span_ray', 'ray.span')

    prob.model.connect('k_WL_manta', 'manta.k_WL')
    prob.model.connect('k_WL_ray', 'ray.k_WL')

    prob.model.connect('lift_manta', 'manta.lift')
    prob.model.connect('lift_ray', 'ray.lift')




    


    


    



    # Setup problem
    prob.setup()


    om.n2(prob)
    
    # Run baseline case
    prob.run_model()

    prob.check_partials(compact_print=True)

