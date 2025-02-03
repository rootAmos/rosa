import openmdao.api as om

import numpy as np

import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.sizing.aero.lift.group_cl0 import GroupCL0
from src.sizing.aero.lift.cl import LiftCoefficient
from src.sizing.aero.lift.group_cla_ray import GroupCLAlphaRay

from src.sizing.aero.lift.ang_attack import AngleOfAttack
from src.sizing.aero.drag.oswald_eff import OswaldEfficiency
from src.sizing.aero.drag.cdi import InducedDrag


class GroupCDiRay(om.Group):
    """

    Group that computes induced drag for a lifting surface.
    Includes zero-lift angle calculation, total lift calculation, 
    Oswald efficiency, and induced drag computation.
    """
    
    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Ray configuration')
        self.options.declare('N', default=1, desc='Number of nodes')

    def setup(self):

        N = self.options['N']

        """
        
        
        self.add_subsystem('cl_alpha', GroupCLAlphaRay(manta=self.options['manta'], N=N),promotes=['*'])

        


        # Add zero-lift angle calculator
        self.add_subsystem('cl0', GroupCL0(N=N),
                          promotes_inputs=['*'],promotes_outputs=['*'])


        

        self.add_subsystem('alpha', AngleOfAttack(N=N), promotes_inputs=['*'],
                     promotes_outputs=['*'])
        

        # Add total lift calculator
        self.add_subsystem('ray_cl',
                        LiftCoefficient(N=N),
                        promotes_inputs=['*'],
                        promotes_outputs=['*'])

        """

        # Add Oswald efficiency calculator
        self.add_subsystem('oswald', OswaldEfficiency(),
                          promotes_inputs=['*'],promotes_outputs=['*'])    # Taper ratio
        

        # Add induced drag calculator
        self.add_subsystem('cdi', InducedDrag(N=N),
                          promotes_inputs=['*'],     # Aspect ratio
                          promotes_outputs=['*'])   # Induced drag coefficient



if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem(reports=False)
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    N = 1
    

    # Wing parameters
    #ivc.add_output('alpha0_airfoil', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    #ivc.add_output('d_twist', val=0.0, units='rad', desc='twist angle')
    #ivc.add_output('CL_alpha_eff', val=5.0, units='1/rad', desc='Wing lift curve slope')
   

    #ivc.add_output('mach', val=0.78 * np.ones(N), desc='Manta Mach number')
    #ivc.add_output('phi_50', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    #ivc.add_output('cl_alpha_airfoil', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')


    #ivc.add_output('d_eps_manta_d_alpha', val=0.25, desc='Manta downwash derivative')
    #ivc.add_output('d_eps_ray_d_alpha', val=0.25, desc='Manta downwash derivative')


    ivc.add_output('aspect_ratio', val=5.0, desc='aspect_ratiospect ratio')
    ivc.add_output('lambda', val=0.45, desc='Taper ratio')
    ivc.add_output('sweep_25', val=12.0, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('h_winglet', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('span', val=30.0, units='m', desc='Wing span')
    ivc.add_output('k_WL', val=2.83, desc='Winglet effectiveness factor')

    #ivc.add_output('lift', val=10000 * np.ones(N), units='N', desc='Lift')
    ivc.add_output('rho', val=0.3639 * np.ones(N), units='kg/m**3', desc='Density')
    #ivc.add_output('u', val=50.0 * np.ones(N), units='m/s', desc='Flow speed')

    ivc.add_output('S_manta', val=100.0, units='m**2', desc='Reference area')
    ivc.add_output('S_ray', val=60.0, units='m**2', desc='Ray reference area')

    ivc.add_output('CL', val=0.2, desc='ray lift coefficient')
    ivc.add_output('CL0', val=0.02, desc='ray zero lift coefficient')


    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cdi_ray', GroupCDiRay(manta=1, N=N), promotes=['*'])


    #prob.model.connect('S_manta', 'S_ref')




    # Setup problem
    prob.setup()

    om.n2(prob.model)
    
    # Run baseline case
    prob.run_model()

    prob.check_partials(compact_print=True)


