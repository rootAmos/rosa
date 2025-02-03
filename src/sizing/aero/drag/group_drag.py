import openmdao.api as om
import numpy as np


import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))



from src.sizing.aero.drag.group_cd_ray import GroupCDRay
from src.sizing.aero.drag.group_cd_manta import GroupCDManta
from src.sizing.aero.drag.group_cd_mantaray import GroupCDMantaRay


class DragForce(om.ExplicitComponent):
    """

    Calculates drag force for a lifting surface (wing or canard).
    

    Inputs:
        rho : float
            Air density [kg/m^3]
        V : float
            Airspeed [m/s]
        CD : float
            Drag coefficient [-]
        S : float
            Reference area [m^2]
    
    Outputs:
        D : float
            Drag force [N]
    """
    
    def setup(self):
        # Flow condition inputs
        self.add_input('rho', val=1.225, units='kg/m**3', desc='Air density')
        self.add_input('u', val=0.0, units='m/s', desc='Airspeed')
        
        # Surface inputs
        self.add_input('CD', val=0.0, desc='Drag coefficient')
        self.add_input('S_ref', val=0.0, units='m**2', desc='Reference area')
        
        # Output
        self.add_output('drag', val=0.0, units='N', desc='Drag force')
        
        # Declare partials
        self.declare_partials('drag', ['rho', 'u', 'CD', 'S_ref'])
        
    def compute(self, inputs, outputs):
        
        # Dynamic pressure
        q = 0.5 * inputs['rho'] * inputs['u']**2
        
        # Drag force
        outputs['drag'] = q * inputs['CD'] * inputs['S_ref']
        
    def compute_partials(self, inputs, partials):

        # Common terms
        dq_drho = 0.5 * inputs['u']**2
        dq_dV = inputs['rho'] * inputs['u']
        
        partials['drag', 'rho'] = dq_drho * inputs['CD'] * inputs['S_ref']
        partials['drag', 'u'] = dq_dV * inputs['CD'] * inputs['S_ref']
        partials['drag', 'CD'] = 0.5 * inputs['rho'] * inputs['u']**2 * inputs['S_ref']
        partials['drag', 'S_ref'] = 0.5 * inputs['rho'] * inputs['u']**2 * inputs['CD']



class GroupDragMantaRay(om.Group):
    """
    Group that computes total drag force by summing contributions
    from wing and canard.
    """

    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')

    def setup(self):
        # Wing drag force

        self.add_subsystem('cd', 
                          GroupCDMantaRay(manta=self.options['manta'],
                                  ray=self.options['ray']),
                          promotes_inputs=['rho','u'],
                          promotes_outputs=['CD'])



        # Canard drag force
        self.add_subsystem('drag',
                          DragForce(),
                          promotes_inputs=['rho','u'],
                          promotes_outputs=[('drag', 'D_canard')])


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Flow conditions
    ivc.add_output('rho', val=1.225, units='kg/m**3', desc='Air density')
    ivc.add_output('u', val=100.0, units='m/s', desc='Airspeed')
    
    # Wing parameters
    ivc.add_output('CD', val=0.7/18, desc='Wing drag coefficient')
    ivc.add_output('S_ref', val=50.0, units='m**2', desc='Wing reference area')
    
    m_opt = 1
    r_opt = 1
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('drag', GroupDragMantaRay(manta=m_opt, ray=r_opt), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()

    
    print('\nBaseline Configuration:')
    print('----------------------')
    print('Flow Conditions:')
    print(f'  Density:             {prob.get_val("rho")[0]:8.3f} kg/m³')
    print(f'  Velocity:            {prob.get_val("V")[0]:8.3f} m/s')
    
    print('\nWing:')
    print(f'  CD:                  {prob.get_val("CD_wing")[0]:8.4f}')
    print(f'  Area:                {prob.get_val("S_wing")[0]:8.3f} m²')
    print(f'  Drag Force:          {prob.get_val("D_wing")[0]/1000:8.3f} kN')
    
    print('\nCanard:')
    print(f'  CD:                  {prob.get_val("CD_canard")[0]:8.4f}')
    print(f'  Area:                {prob.get_val("S_canard")[0]:8.3f} m²')
    print(f'  Drag Force:          {prob.get_val("D_canard")[0]/1000:8.3f} kN')
    
    print('\nTotal:')
    print(f'  Drag Force:          {prob.get_val("D_total")[0]/1000:8.3f} kN')
    
    # Parameter sweeps
    import matplotlib.pyplot as plt
    

    prob.check_partials(compact_print=True)
    
    plt.tight_layout()
    plt.show() 