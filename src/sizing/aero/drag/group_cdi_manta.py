import openmdao.api as om
from oswald_eff import OswaldEfficiency
from cdi import InducedDrag

import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.sizing.aero.lift.group_cl0 import GroupCL0
from src.sizing.aero.lift.cl import LiftCoefficient
from src.sizing.aero.lift.group_cla_manta import GroupCLAlphaManta

from src.sizing.aero.lift.ang_attack import AngleOfAttack
import numpy as np



class GroupCDiManta(om.Group):
    """

    Group that computes induced drag for a lifting surface.
    Includes zero-lift angle calculation, total lift calculation, 
    Oswald efficiency, and induced drag computation.
    """
    
    def initialize(self):
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')

    def setup(self):

        self.add_subsystem('cl_alpha', GroupCLAlphaManta(ray=self.options['ray']),promotes=['*'])

        
        # Add zero-lift angle calculator
        self.add_subsystem('cl0', GroupCL0(),
                          promotes_inputs=['*'],promotes_outputs=['*'])

        

        self.add_subsystem('alpha', AngleOfAttack(), promotes_inputs=['*'],
                     promotes_outputs=['*'])
        
        # Add total lift calculator
        self.add_subsystem('manta_cl',
                        LiftCoefficient(),
                        promotes_inputs=['*'],
                        promotes_outputs=['*'])
        

        # Add Oswald efficiency calculator
        self.add_subsystem('oswald', OswaldEfficiency(),
                          promotes_inputs=['*'],promotes_outputs=['*'])    # Taper ratio
        
        # Add induced drag calculator
        self.add_subsystem('cdi', InducedDrag(),
                          promotes_inputs=['*'],     # Aspect ratio
                          promotes_outputs=['*'])   # Induced drag coefficient


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem(reports=False)
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Wing parameters
    ivc.add_output('alpha0_airfoil', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_twist', val=0.0, units='rad', desc='twist angle')
    #ivc.add_output('CL_alpha_eff', val=5.0, units='1/rad', desc='Wing lift curve slope')
   
    ivc.add_output('mach', val=0.78, desc='Manta Mach number')
    ivc.add_output('phi_50', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')

    ivc.add_output('d_eps_manta_d_alpha', val=0.25, desc='Manta downwash derivative')
    ivc.add_output('d_eps_ray_d_alpha', val=0.25, desc='Manta downwash derivative')

    ivc.add_output('aspect_ratio', val=5.0, desc='aspect_ratiospect ratio')
    ivc.add_output('taper', val=0.45, desc='Taper ratio')
    ivc.add_output('sweep_25', val=12.0, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('h_winglet', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('span', val=30.0, units='m', desc='Wing span')
    ivc.add_output('k_WL', val=2.83, desc='Winglet effectiveness factor')

    ivc.add_output('lift', val=10000, units='N', desc='Lift')
    ivc.add_output('rho', val=0.3639, units='kg/m**3', desc='Density')
    ivc.add_output('u', val=50.0, units='m/s', desc='Flow speed')
    ivc.add_output('S_ref', val=100.0, units='m**2', desc='Reference area')



    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cdi_manta', GroupCDiManta(ray=1), promotes=['*'])


    # Setup problem
    prob.setup()

    om.n2(prob.model)
    
    # Run baseline case
    prob.run_model()

    prob.check_partials(compact_print=True)


