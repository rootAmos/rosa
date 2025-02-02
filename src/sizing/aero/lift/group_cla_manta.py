import openmdao.api as om
import numpy as np

import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


from src.sizing.aero.lift.cl_alpha_airfoil import LiftCurveSlope3D

class CoupledCLAlphaManta(om.ExplicitComponent):

    """
    Calculates the effective lift curve slope for the wing, accounting for downwash.
    

    Inputs:
        CL_alpha : float
            Manta lift curve slope [1/rad]
        deps_manta_da : float
            Change in wing downwash angle with respect to angle of attack [-]
    
    Outputs:
        CL_alpha_eff : float
            Effective wing lift curve slope [1/rad]
    """

    def initialize(self):
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
    
    def setup(self):

        # Inputs
        self.add_input('CL_alpha', val=0.0, units='1/rad', 
                      desc='Manta lift curve slope')
        
        if self.options['ray']:
            self.add_input('d_eps_manta_d_alpha', val=0.0, 
                      desc='Manta downwash derivative from canard influence')
        # end
        
        # Outputs
        self.add_output('CL_alpha_eff', val=0.0, units='1/rad',
                       desc='Effective wing lift curve slope')
        
        # Partials
        self.declare_partials('CL_alpha_eff', ['*'])
        

    def compute(self, inputs, outputs):
        CL_alpha = inputs['CL_alpha']
        d_eps_manta_d_alpha = inputs['d_eps_manta_d_alpha']

        if self.options['ray'] == 1:
            outputs['CL_alpha_eff'] = CL_alpha * (1 - d_eps_manta_d_alpha)
        else:
            outputs['CL_alpha_eff'] = CL_alpha

        # end
        

    def compute_partials(self, inputs, partials):
        partials['CL_alpha_eff', 'CL_alpha'] = 1 - inputs['d_eps_manta_d_alpha']
        partials['CL_alpha_eff', 'd_eps_manta_d_alpha'] = -inputs['CL_alpha'] 


class GroupCLAlphaManta(om.Group):
    """
    Group that computes total lift curve slope by summing contributions
    from wing and canard.
    """

    def initialize(self):
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
    
    def setup(self):
      
        self.add_subsystem('cl_alpha_3d',
            LiftCurveSlope3D(),
            promotes_inputs=['*'],
            promotes_outputs=['*'])
                    

        self.add_subsystem('cpld_cl_alpha', 
                        CoupledCLAlphaManta(ray=self.options['ray']),
                        promotes_inputs=['*'],
                        promotes_outputs=['*'])
        # end



if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
        
    # Manta parameters
    ivc.add_output('mach', val=0.78, desc='Manta Mach number')
    ivc.add_output('phi_50', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')

    ivc.add_output('aspect_ratio', val=10.0, desc='Manta aspect ratio')
    ivc.add_output('d_eps_manta_d_alpha', val=0.25, desc='Manta downwash derivative')
    

    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cl_alpha_coupled', GroupCLAlphaManta(ray=1), promotes=['*'])


    
    # Setup problem
    prob.setup()
    
    om.n2(prob.model)
    # Run baseline case
    prob.run_model()



    
    print('\nBaseline Configuration:')
    print('----------------------')
    print('Manta:')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha")[0]:8.3f} /rad')
    print(f'  Downwash:            {prob.get_val("d_eps_manta_d_alpha")[0]:8.3f}')
    print(f'  Effective CL_alpha:   {prob.get_val("CL_alpha_eff")[0]:8.3f} /rad')
    
    print('\nCanard:')

    prob.check_partials(compact_print=True)