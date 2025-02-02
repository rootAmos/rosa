import openmdao.api as om
import numpy as np

from cl_alpha_airfoil import LiftCurveSlope3D

class CoupledCLAlphaRay(om.ExplicitComponent):
    """
    Calculates the effective lift curve slope for the canard, accounting for area ratio and downwash.
    

    Inputs:
        CL_alpha : float
            Ray lift curve slope [1/rad]
        deps_ray_da : float
            Change in canard downwash angle with respect to angle of attack [-]
        S_ray : float
            Ray reference area [m^2]
        S_manta : float
            Manta reference area [m^2]
    
    Outputs:
        CL_alpha_eff : float
            Effective canard lift curve slope [1/rad]
    """

    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
    

    def setup(self):
        # Inputs

        self.add_input('CL_alpha', val=0.0, units='1/rad',
                      desc='Ray lift curve slope')

        
        if self.options['manta']:
            self.add_input('S_ray', val=0.0, units='m**2',
                          desc='Ray reference area')
            self.add_input('S_manta', val=0.0, units='m**2',
                          desc='Manta reference area')
            self.add_input('d_eps_ray_d_alpha', val=0.0,
                      desc='Ray downwash derivative')
        # end

        
        # Outputs
        self.add_output('CL_alpha_eff', val=0.0, units='1/rad',
                       desc='Effective canard lift curve slope')

        
        # Partials
        self.declare_partials('CL_alpha_eff', ['*'])
        
    def compute(self, inputs, outputs):
        CL_alpha = inputs['CL_alpha']
        d_eps_ray_d_alpha = inputs['d_eps_ray_d_alpha']

        if self.options['manta']:
            S_ray = inputs['S_ray']
            S_manta = inputs['S_manta']
            area_ratio = S_ray / S_manta
            outputs['CL_alpha_eff'] = CL_alpha  * (1 + d_eps_ray_d_alpha) * area_ratio
        else:
            outputs['CL_alpha_eff'] = CL_alpha 
        # end

        
    def compute_partials(self, inputs, partials):
        d_eps_ray_d_alpha = inputs['d_eps_ray_d_alpha']
        CL_alpha = inputs['CL_alpha']

        if self.options['manta'] ==1:
            S_ray = inputs['S_ray']
            S_manta = inputs['S_manta']
            area_ratio = S_ray / S_manta
            partials['CL_alpha_eff', 'CL_alpha'] = (1 + d_eps_ray_d_alpha) * area_ratio
            partials['CL_alpha_eff', 'd_eps_ray_d_alpha'] = CL_alpha  * area_ratio
            partials['CL_alpha_eff', 'S_ray'] = CL_alpha * (1 + d_eps_ray_d_alpha) / S_manta
            partials['CL_alpha_eff', 'S_manta'] = -CL_alpha * (1 + d_eps_ray_d_alpha) * S_ray / S_manta**2
        else:


            partials['CL_alpha_eff', 'CL_alpha'] = 1


class GroupCLAlphaRay(om.Group):
    """
    Group that computes total lift curve slope by summing contributions
    from wing and canard.
    """

    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')

    def setup(self):


        self.add_subsystem('cl_alpha_3d',
                            LiftCurveSlope3D(),
                            promotes_inputs=['*'],
                            promotes_outputs=['*'])
        
        # Ray lift curve slope
        self.add_subsystem('cpld_cl_alpha',
                        CoupledCLAlphaRay(manta=self.options['manta']),
                        promotes_inputs=['*'],
                        promotes_outputs=['*'])

    # end

    

if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Ray parameters
    ivc.add_output('mach', val=0.78, desc='Ray Mach number')
    ivc.add_output('phi_50', val=4.0, units='deg', desc='Ray 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil', val=2*np.pi, units='1/rad', desc='Ray airfoil lift curve slope')

    ivc.add_output('aspect_ratio', val=10.0, desc='Ray aspect ratio')
    ivc.add_output('d_eps_ray_d_alpha', val=0.1, desc='Ray downwash derivative')
    
    S_ref = 25
    ivc.add_output('S_ray', val=S_ref, units='m**2', desc='Ray wing area')
    ivc.add_output('S_manta', val=S_ref, units='m**2', desc='Manta wing area')
    


    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cl_alpha_ray', GroupCLAlphaRay(manta=1), promotes=['*'])




    
    # Setup problem
    prob.setup()
    
    om.n2(prob.model)
    # Run baseline case
    prob.run_model()



    
    print('\nBaseline Configuration:')
    print('----------------------')
    print('Manta:')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha")[0]:8.3f} /rad')
    print(f'  Downwash:            {prob.get_val("d_eps_ray_d_alpha")[0]:8.3f}')

    print(f'  Effective CL_alpha:   {prob.get_val("CL_alpha_eff")[0]:8.3f} /rad')
    


    print('\nRay:')

    prob.check_partials(compact_print=True)