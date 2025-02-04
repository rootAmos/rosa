import openmdao.api as om
import numpy as np

import os
import sys
import pdb
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.sizing.aero.lift.group_cl0 import GroupCL0
from src.sizing.aero.lift.group_cla_manta import GroupCLAlphaManta
from src.sizing.aero.lift.lift_req import LiftRequired
from src.sizing.aero.lift.ang_attack import AngleOfAttack
from src.sizing.aero.lift.cl import LiftCoefficient



class GroupCLManta(om.Group):
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
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
        self.options.declare('N', default=1, desc='Number of nodes')
    
    def setup(self):


        N = self.options['N']
        # Step 3: Lift curve slopes
        self.add_subsystem('cl_alpha',
                            GroupCLAlphaManta(ray=self.options['ray'], N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
        

        # Step 2: Zero-angle lift coefficients
        self.add_subsystem('cl0',
                          GroupCL0(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
        


        self.add_subsystem('alpha', AngleOfAttack(N=N), promotes_inputs=['*'],
                     promotes_outputs=['*'])
        


        # Step 4: Individual lift coefficients
        self.add_subsystem('cl',
                          LiftCoefficient(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])


        


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()

    N = 1
    
    # Flight conditions
    ivc.add_output('lift', val=10000.0 * np.ones(N), units='N', desc='Required lift force')
    ivc.add_output('rho', val=0.38 * np.ones(N), units='kg/m**3', desc='Air density')



    ivc.add_output('u', val=70.0 * np.ones(N), units='m/s', desc='Flight speed')
    ivc.add_output('S_ref', val=100.0, units='m**2', desc='Reference area')

    # Manta parameters
    ivc.add_output('mach', val=0.78 * np.ones(N), desc='Manta Mach number')
    ivc.add_output('phi_50', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')

    ivc.add_output('aspect_ratio', val=10.0, desc='Manta aspect ratio')
    #ivc.add_output('d_eps_manta_d_alpha', val=0.25, desc='Manta downwash derivative')
    
    # Wing parameters
    ivc.add_output('alpha0_airfoil', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_twist', val=0.0, units='rad', desc='twist angle')

    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cl', GroupCLManta(ray=0, N=N), promotes=['*'])





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
    
    print('\nManta:')
    print(f'  CL0:                 {prob.get_val("CL0")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha_eff")[0]:8.3f} /rad')
    print(f'  CL:                  {prob.get_val("CL")[0]:8.3f}')

    prob.check_partials(compact_print=True)    
    