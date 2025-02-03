import openmdao.api as om
import numpy as np

import os
import sys
import pdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.sizing.aero.lift.group_cl_manta import GroupCLManta
from src.sizing.aero.drag.group_cd_manta import GroupCDManta
from src.sizing.mission.turbo_cruise import ComputeTurboCruiseRange, PropulsiveEfficiency
from src.sizing.ptrain.ductedfan import ComputeDuctedFan
from src.sizing.ptrain.ductedfan import ComputeUnitThrust

from src.sizing.aero.lift.lift_req import LiftRequired
from src.sizing.aero.drag.drag_force import DragForce

from src.sizing.mission.mach_number import MachNumber
from src.sizing.mission.atmos import ComputeAtmos


class Weight(om.ExplicitComponent):
    """

    Computes weight (force) from mass using W = mg
    """
    
    def initialize(self):
        self.options.declare('g', default=9.806, desc='Gravitational acceleration (m/s^2)')
        
    def setup(self):
        # Inputs
        self.add_input('mass', val=1.0, units='kg',
                      desc='Mass of object')
        
        # Outputs
        self.add_output('weight', val=1.0, units='N',
                      desc='Weight (force) of object')
        
        # Declare partials
        self.declare_partials('weight', 'mass')
        
    def compute(self, inputs, outputs):
        g = self.options['g']
        
        outputs['weight'] = inputs['mass'] * g
        
    def compute_partials(self, inputs, partials):
        g = self.options['g']
        
        partials['weight', 'mass'] = g



class CruiseRequirements(om.Group):
    """

    Group that computes cruise requirements:
    1. Required lift force
    2. Lift coefficient from Manta configuration
    3. Drag coefficient from Manta configuration
    4. Required drag force
    5. Unit thrust distribution
    6. Ducted fan performance
    7. Total propulsive efficiency
    8. Cruise range based on turbogenerator performance
    """
    
    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')
        self.options.declare('manta', default=1)
        self.options.declare('ray', default=0)  # No Ray in cruise
    
    def setup(self):
        N = self.options['N']

        self.add_subsystem('atmos',
                          ComputeAtmos(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

        # Add Mach number computation
        self.add_subsystem('mach',
                          MachNumber(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])


        # Add lift requirement computation
        self.add_subsystem('lift_req',
                          LiftRequired(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])


        # Add lift coefficient computation
        self.add_subsystem('lift',
                          GroupCLManta( 
                                        ray=self.options['ray'],
                                        N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

        # Add drag coefficient computation
        self.add_subsystem('drag',
                          GroupCDManta(
                                        ray=self.options['ray'],
                                        N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

        # Add drag force computation
        self.add_subsystem('drag_force',
                          DragForce(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

        # Add unit thrust computation
        self.add_subsystem('unit_thrust',
                          ComputeUnitThrust(),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

        # Add ducted fan computation
        self.add_subsystem('ductedfan',
                          ComputeDuctedFan(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

        # Add propulsive efficiency calculation
        self.add_subsystem('prop_eff',
                          PropulsiveEfficiency(),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
        
        self.add_subsystem('weight_mto', Weight(g=9.806), promotes_inputs=[], promotes_outputs=[])
        self.add_subsystem('weight_fuel', Weight(g=9.806), promotes_inputs=[], promotes_outputs=[])


        # Add cruise range computation
        self.add_subsystem('range',
                          ComputeTurboCruiseRange(),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

        # Connect lift and drag coefficients
        self.connect('CL', 'cl')
        self.connect('CD_manta', ['cd','CD'])
        
        # Connect thrust distribution
        self.connect('weight_mto.weight', 'w_mto')
        self.connect('weight_fuel.weight', 'w_fuel')

        self.connect('drag', 'thrust_total')



if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem(reports=False)
    
    # Independent variable component
    ivc = om.IndepVarComp()
    
    N = 1
    manta = 1
    ray = 0
    
    # Flight conditions
    ivc.add_output('alt', val=30000 * np.ones(N), units='ft')

    ivc.add_output('u', val=230.0 * np.ones(N), units='m/s')
    ivc.add_output('gamma', val=0.0 * np.ones(N), units='deg')
    ivc.add_output('q', val=21000.0 * np.ones(N), units='Pa')
    
    # Manta parameters
    ivc.add_output('S_manta', val=100.0, units='m**2')
    ivc.add_output('aspect_ratio_manta', val=10.0)
    ivc.add_output('sweep_25_manta', val=15.0, units='deg')
    ivc.add_output('taper_manta', val=0.45)
    ivc.add_output('mto', val=55000.0, units='kg')
    ivc.add_output('m_fuel', val=600.0, units='kg')
    
    # Thrust distribution parameters
    ivc.add_output('n_fans', val=2)
    
    # Ducted fan parameters
    ivc.add_output('d_blades', val=1.5, units='m')
    ivc.add_output('d_hub', val=0.5, units='m')
    ivc.add_output('epsilon_r', val=1.5)
    ivc.add_output('eta_fan', val=0.95)
    ivc.add_output('eta_duct', val=0.96)
    
    # Propulsion system parameters
    ivc.add_output('eta_pe', val=0.95)
    ivc.add_output('eta_motor', val=0.95)
    ivc.add_output('eta_gen', val=0.95)
    ivc.add_output('cp', val=0.5/60/60/1000, units='kg/W/s')
    
    # Mission parameters
    ivc.add_output('target_range', val=1000.0, units='km')


    #ivc.add_output('d_eps_manta_d_alpha', val=0.25, desc='Manta downwash derivative')
    
    # Wing parameters
    ivc.add_output('alpha0_airfoil', val=-2.0, units='deg', desc='Wing airfoil zero-lift angle')
    ivc.add_output('d_twist', val=0.0, units='rad', desc='twist angle')


   
    ivc.add_output('phi_50', val=5.0, units='deg', desc='Manta 50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil', val=2*np.pi, units='1/rad', desc='Manta airfoil lift curve slope')


    ivc.add_output('aspect_ratio', val=5.0, desc='aspect ratio')
    ivc.add_output('lambda', val=0.45, desc='Taper ratio')
    ivc.add_output('sweep_25', val=12.0, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('h_winglet', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('span', val=30.0, units='m', desc='Wing span')
    ivc.add_output('k_WL', val=2.83, desc='Winglet effectiveness factor')

    ivc.add_output('Q_manta', val=1.1, desc='Interference factor')
    ivc.add_output('t_c', val=0.19, desc='Thickness to chord ratio')

    ivc.add_output('tau', val=0.8, desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
    ivc.add_output('k_lam_manta', val=0.1, desc='Laminar flow fraction')
    ivc.add_output('sweep_max_t', val=10, units='deg', desc='Wing sweep at maximum thickness')
    ivc.add_output('l_char_manta', val=1.0, units='m', desc='Characteristic length')

    ivc.add_output('mu', val=1.789e-5 * np.ones(N), units='Pa*s', desc='Dynamic viscosity')

    ivc.add_output('c_duct', val=3.0, units='m', desc='Nacelle length')
    ivc.add_output('od_duct', val=2.0, units='m', desc='Duct outer diameter')
    ivc.add_output('id_duct', val=1.0, units='m', desc='Duct inner diameter')
    ivc.add_output('Q_duct', val=1.0, desc='Duct interference factor')
    ivc.add_output('k_lam_duct', val=0.1, desc='Duct laminar flow fraction')
    ivc.add_output('num_ducts', val=2, desc='Number of ducts')
    
    # Build the model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cruise', 
                            CruiseRequirements(N=N, manta=manta, ray=ray),
                            promotes=['*'])
    
    prob.model.connect('S_manta', ['S_ref','S_exp'])
    prob.model.connect('mto', 'weight_mto.mass')
    prob.model.connect('m_fuel', 'weight_fuel.mass')
    prob.model.connect('Q_duct', 'ducts.Q')
    prob.model.connect('Q_manta', 'wing.Q')
    prob.model.connect('k_lam_manta', 'wing.k_lam')
    prob.model.connect('k_lam_duct', 'ducts.k_lam')
    prob.model.connect('l_char_manta', 'wing.l_char')
    prob.model.connect('c_duct', 'ducts.l_char')










    # Setup problem
    prob.setup()

    #om.n2(prob.model)
    
    # Run model
    prob.run_model()
    
    # Print results
    print("\nCruise Requirements Results:")
    print("--------------------------")
    print(f"CL: {prob.get_val('CL')[0]:.3f}")
    print(f"CD: {prob.get_val('CD_manta')[0]:.3f}")

    print(f"L/D: {prob.get_val('CL')[0]/prob.get_val('CD_manta')[0]:.1f}")
    print(f"Total thrust required: {prob.get_val('drag')[0]/1000:.1f} kN")
    print(f"Thrust per fan: {prob.get_val('thrust_unit')[0]/1000:.1f} kN")

    print(f"Power per fan: {prob.get_val('p_shaft_unit')[0]/1000:.1f} kW")
    print(f"Propulsive efficiency: {prob.get_val('eta_pt')[0]:.3f}")
    print(f"Range: {prob.get_val('range')[0]/1000:.1f} km")
    
    # Check partials
    prob.check_partials(compact_print=True) 