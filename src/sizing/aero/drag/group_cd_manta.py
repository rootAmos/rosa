import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt


import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))



from src.sizing.aero.drag.group_cdi_manta import GroupCDiManta
from src.sizing.aero.drag.group_cd0_manta import GroupCD0Manta
from src.sizing.aero.drag.wave_drag import WaveDrag

class GroupCDManta(om.Group):
    """
    Group that computes total induced drag coefficient by summing contributions
    from wing and canard.
    """
    def initialize(self):

        self.options.declare('ray', default=0, desc='Flag for Ray configuration')


    def setup(self):
        # Wing induced drag

        self.add_subsystem('cdi_manta', GroupCDiManta(ray=1), promotes_inputs=['*'], promotes_outputs=['*'])   
        self.add_subsystem('cd0_manta', GroupCD0Manta(), promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('wave_drag', WaveDrag(), promotes_inputs=['*'], promotes_outputs=['*'])
        # end
        




        # Sum the contributions
        adder = om.AddSubtractComp()
        adder.add_equation('CD_manta',
                        ['CD0_manta', 'CDi_manta', 'CD_wave'],
                        desc='Total drag coefficient')

        self.add_subsystem('sum', adder, promotes=['*'])
        # end

        self.connect('sum.CD0', 'CD0_manta')
        self.connect('CDi', 'CDi_manta')




if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Wing parameters
    ivc.add_output('alpha0_airfoil', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_twist', val=0.0, units='rad', desc='twist angle')
   
    ivc.add_output('mach', val=0.78, desc='Mach number')
    ivc.add_output('phi_50', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')

    ivc.add_output('d_eps_manta_d_alpha', val=0.25, desc='Manta downwash derivative')
    ivc.add_output('d_eps_ray_d_alpha', val=0.25, desc='Manta downwash derivative')

    ivc.add_output('aspect_ratio', val=5.0, desc='aspect ratio')
    ivc.add_output('taper', val=0.45, desc='Taper ratio')
    ivc.add_output('sweep_25', val=12.0, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('h_winglet', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('span', val=30.0, units='m', desc='Wing span')
    ivc.add_output('k_WL', val=2.83, desc='Winglet effectiveness factor')

    ivc.add_output('lift', val=10000, units='N', desc='Lift')
    ivc.add_output('rho', val=0.3639, units='kg/m**3', desc='Density')
    ivc.add_output('u', val=50.0, units='m/s', desc='Flow speed')
    ivc.add_output('S_ref', val=100.0, units='m**2', desc='Reference area')

    ivc.add_output('Q_manta', val=1.1, desc='Interference factor')
    ivc.add_output('S_exp', val=50.0, units='m**2', desc='Exposed planform area')
    ivc.add_output('t_c', val=0.19, desc='Thickness to chord ratio')

    ivc.add_output('tau', val=0.8, desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
    ivc.add_output('lambda_w', val=0.45, desc='Wing taper ratio')
    ivc.add_output('k_lam_manta', val=0.1, desc='Laminar flow fraction')
    ivc.add_output('sweep_max_t', val=10, units='deg', desc='Wing sweep at maximum thickness')
    ivc.add_output('l_char_manta', val=1.0, units='m', desc='Characteristic length')

    ivc.add_output('alt', val=30000, units='ft', desc='Altitude')
    ivc.add_output('mu', val=1.789e-5, units='Pa*s', desc='Dynamic viscosity')

    ivc.add_output('c_duct', val=3.0, units='m', desc='Nacelle length')
    ivc.add_output('od_duct', val=2.0, units='m', desc='Duct outer diameter')
    ivc.add_output('id_duct', val=1.0, units='m', desc='Duct inner diameter')
    ivc.add_output('Q_duct', val=1.0, desc='Duct interference factor')
    ivc.add_output('k_lam_duct', val=0.1, desc='Duct laminar flow fraction')
    ivc.add_output('num_ducts', val=2, desc='Number of ducts')

    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('CD', GroupCDManta(), promotes=['*'])
    
    prob.model.connect('l_char_manta', 'wing.l_char')
    prob.model.connect('c_duct', 'ducts.l_char')

    prob.model.connect('k_lam_manta', 'wing.k_lam')
    prob.model.connect('k_lam_duct', 'ducts.k_lam')


    prob.model.connect('Q_duct', 'wing.Q')
    prob.model.connect('Q_manta', 'ducts.Q')


    # Setup problem
    prob.setup()


    om.n2(prob)
    
    # Run baseline case
    prob.run_model()

    prob.check_partials(compact_print=True)

