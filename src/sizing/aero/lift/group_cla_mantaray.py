import openmdao.api as om
import numpy as np

from cl_alpha_airfoil import LiftCurveSlope3D
from group_cla_manta import GroupCLAlphaManta
from group_cla_ray import GroupCLAlphaRay

class GroupCLAlphaMantaRay(om.Group):
    """
    Group that computes total lift curve slope by summing contributions
    from wing and canard.
    """

    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
    
    def setup(self):
        # Manta lift curve slope

        self.add_subsystem('cl_alpha_manta', 
                        GroupCLAlphaManta(ray=self.options['ray']),
                        promotes_inputs=['d_eps_manta_d_alpha'],
                        promotes_outputs=[])
    

        self.add_subsystem('cl_alpha_ray', 
                        GroupCLAlphaRay(manta=self.options['manta']),
                        promotes_inputs=['S_ray', 'S_manta','d_eps_ray_d_alpha'],
                        promotes_outputs=[])

        adder = om.AddSubtractComp()
        adder.add_equation('CL_alpha_total',
                        ['CL_alpha_manta_eff', 'CL_alpha_ray_eff'],
                        desc='Total lift curve slope')
        self.add_subsystem('sum_CL_alpha', adder, promotes=['*'])

        self.connect('cl_alpha_manta.CL_alpha_eff', 'CL_alpha_manta_eff')
        self.connect('cl_alpha_ray.CL_alpha_eff', 'CL_alpha_ray_eff')
        # end



if __name__ == "__main__":

    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()

    # Manta parameters
    ivc.add_output('mach_manta', val=0.78, desc='Manta Mach number')
    ivc.add_output('phi_50_manta', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil_manta', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')
    ivc.add_output('aspect_ratio_manta', val=10.0, desc='Manta aspect ratio')
    ivc.add_output('d_eps_manta_d_alpha', val=0.25, desc='Manta downwash derivative')

    # Ray parameters
    ivc.add_output('mach_ray', val=0.78, desc='Ray Mach number')
    ivc.add_output('phi_50_ray', val=4.0, units='deg', desc='Ray 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil_ray', val=2*np.pi, units='1/rad', desc='Ray airfoil lift curve slope')
    ivc.add_output('aspect_ratio_ray', val=10.0, desc='Ray aspect ratio')
    ivc.add_output('d_eps_ray_d_alpha', val=0.1, desc='Ray downwash derivative')
    
    ivc.add_output('S_ray', val=20.0, units='m**2', desc='Ray reference area')
    ivc.add_output('S_manta', val=120.0, units='m**2', desc='Manta reference area')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cl_alpha_mantaray', GroupCLAlpha(manta=1, ray=1), promotes=['*'])

    
    prob.model.connect('mach_manta', 'cl_alpha_manta.mach')
    prob.model.connect('phi_50_manta', 'cl_alpha_manta.phi_50')
    prob.model.connect('aspect_ratio_manta', 'cl_alpha_manta.aspect_ratio')
    prob.model.connect('cl_alpha_airfoil_manta', 'cl_alpha_manta.cl_alpha_airfoil')


    prob.model.connect('mach_ray', 'cl_alpha_ray.mach')
    prob.model.connect('phi_50_ray', 'cl_alpha_ray.phi_50')
    prob.model.connect('cl_alpha_airfoil_ray', 'cl_alpha_ray.cl_alpha_airfoil')
    prob.model.connect('aspect_ratio_ray', 'cl_alpha_ray.aspect_ratio')




    
    # Setup problem
    prob.setup()
    
    om.n2(prob.model)
    # Run baseline case
    prob.run_model()



    
    print('\nBaseline Configuration:')
    print('----------------------')
    print('Manta:')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha_total")[0]:8.3f} /rad')
    print(f'  Downwash:            {prob.get_val("CL_alpha_manta_eff")[0]:8.3f}')
    print(f'  Effective CL_alpha:   {prob.get_val("CL_alpha_ray_eff")[0]:8.3f} /rad')
    

    print('\nRay:')

    prob.check_partials(compact_print=True)