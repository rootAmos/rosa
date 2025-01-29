import openmdao.api as om
import numpy as np

class LiftForce(om.ExplicitComponent):
    """
    Calculates lift forces for wing and canard, with canard dynamic pressure ratio.
    
    Inputs:
        rho : float
            Air density [kg/m^3]
        V : float
            Airspeed [m/s]
        CL_w : float
            Wing lift coefficient [-]
        S_w : float
            Wing reference area [m^2]
        CL_c : float
            Canard lift coefficient [-]
        S_c : float
            Canard reference area [m^2]
        eta_c : float
            Canard dynamic pressure ratio [-]
    
    Outputs:
        L_w : float
            Wing lift force [N]
        L_c : float
            Canard lift force [N]
        L : float
            Total lift force [N]
    """
    
    def setup(self):
        # Flow condition inputs
        self.add_input('rho', val=1.225, units='kg/m**3', desc='Air density')
        self.add_input('V', val=0.0, units='m/s', desc='Airspeed')
        
        # Wing inputs
        self.add_input('CL_w', val=0.0, desc='Wing lift coefficient')
        self.add_input('S_w', val=0.0, units='m**2', desc='Wing reference area')
        
        # Canard inputs
        self.add_input('CL_c', val=0.0, desc='Canard lift coefficient')
        self.add_input('S_c', val=0.0, units='m**2', desc='Canard reference area')
        self.add_input('eta_c', val=1.0, desc='Canard dynamic pressure ratio')
        
        # Outputs
        self.add_output('L_w', val=0.0, units='N', desc='Wing lift force')
        self.add_output('L_c', val=0.0, units='N', desc='Canard lift force')
        self.add_output('L', val=0.0, units='N', desc='Total lift force')
        
        # Declare partials
        self.declare_partials(['L_w', 'L_c', 'L'], ['rho', 'V', 'CL_w', 'S_w', 'CL_c', 'S_c', 'eta_c'])
        
    def compute(self, inputs, outputs):
        rho = inputs['rho']
        V = inputs['V']
        CL_w = inputs['CL_w']
        S_w = inputs['S_w']
        CL_c = inputs['CL_c']
        S_c = inputs['S_c']
        eta_c = inputs['eta_c']
        
        # Dynamic pressure
        q = 0.5 * rho * V**2
        
        # Individual lift forces
        outputs['L_w'] = q * CL_w * S_w
        outputs['L_c'] = q * CL_c * S_c * eta_c
        
        # Total lift
        outputs['L'] = outputs['L_w'] + outputs['L_c']
        
    def compute_partials(self, inputs, partials):
        rho = inputs['rho']
        V = inputs['V']
        CL_w = inputs['CL_w']
        S_w = inputs['S_w']
        CL_c = inputs['CL_c']
        S_c = inputs['S_c']
        eta_c = inputs['eta_c']
        
        # Common terms
        dq_drho = 0.5 * V**2
        dq_dV = rho * V
        
        # Wing derivatives
        partials['L_w', 'rho'] = dq_drho * CL_w * S_w
        partials['L_w', 'V'] = dq_dV * CL_w * S_w
        partials['L_w', 'CL_w'] = 0.5 * rho * V**2 * S_w
        partials['L_w', 'S_w'] = 0.5 * rho * V**2 * CL_w
        partials['L_w', 'CL_c'] = 0.0
        partials['L_w', 'S_c'] = 0.0
        partials['L_w', 'eta_c'] = 0.0
        
        # Canard derivatives
        partials['L_c', 'rho'] = dq_drho * CL_c * S_c * eta_c
        partials['L_c', 'V'] = dq_dV * CL_c * S_c * eta_c
        partials['L_c', 'CL_c'] = 0.5 * rho * V**2 * S_c * eta_c
        partials['L_c', 'S_c'] = 0.5 * rho * V**2 * CL_c * eta_c
        partials['L_c', 'eta_c'] = 0.5 * rho * V**2 * CL_c * S_c
        partials['L_c', 'CL_w'] = 0.0
        partials['L_c', 'S_w'] = 0.0
        
        # Total lift derivatives
        for var in ['rho', 'V', 'CL_w', 'S_w', 'CL_c', 'S_c', 'eta_c']:
            partials['L', var] = partials['L_w', var] + partials['L_c', var] 