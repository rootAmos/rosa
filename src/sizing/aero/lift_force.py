import openmdao.api as om
import numpy as np

class LiftForce(om.ExplicitComponent):
    """
    Calculates lift force for a lifting surface (wing or canard).
    
    Parameters
    ----------
    surface_type : str
        Type of lifting surface ('wing' or 'canard')
    
    Inputs
    -------
    rho : float
        Air density [kg/m^3]
    V : float
        Airspeed [m/s]
    CL : float
        Lift coefficient [-]
    S : float
        Reference area [m^2]
    eta : float
        Dynamic pressure ratio [-] (only used for canard)
    
    Outputs
    -------
    L : float
        Lift force [N]
    """
    
    def initialize(self):
        self.options.declare('surface_type', default='wing',
                           values=['wing', 'canard'],
                           desc='Type of lifting surface')
    
    def setup(self):
        # Flow condition inputs
        self.add_input('rho', val=1.225, units='kg/m**3', desc='Air density')
        self.add_input('u', val=0.0, units='m/s', desc='Airspeed in the x-direction of the body axis')
        
        # Surface inputs
        self.add_input('CL', val=0.0, desc='Lift coefficient')
        self.add_input('s_ref', val=0.0, units='m**2', desc='Reference area')
        
        # Add eta input only for canard
        if self.options['surface_type'] == 'canard':
            self.add_input('eta', val=1.0, desc='Dynamic pressure ratio')
        
        # Output
        self.add_output('lift', val=0.0, units='N', desc='Lift force')
        
        # Declare partials
        if self.options['surface_type'] == 'wing':
            self.declare_partials('lift', ['rho', 'u', 'CL', 's_ref'])
        else:
            self.declare_partials('lift', ['rho', 'u', 'CL', 's_ref', 'eta'])
        
    def compute(self, inputs, outputs):
        # Dynamic pressure
        q = 0.5 * inputs['rho'] * inputs['u']**2
        
        # Lift force
        if self.options['surface_type'] == 'wing':
            outputs['lift'] = q * inputs['CL'] * inputs['s_ref']
        else:
            outputs['lift'] = q * inputs['CL'] * inputs['s_ref'] * inputs['eta']
        
    def compute_partials(self, inputs, partials):
        # Common terms
        dq_drho = 0.5 * inputs['u']**2
        dq_dV = inputs['rho'] * inputs['u']
        
        if self.options['surface_type'] == 'wing':
            partials['lift', 'rho'] = dq_drho * inputs['CL'] * inputs['s_ref']
            partials['lift', 'u'] = dq_dV * inputs['CL'] * inputs['s_ref']
            partials['lift', 'CL'] = 0.5 * inputs['rho'] * inputs['u']**2 * inputs['s_ref']
            partials['lift', 's_ref'] = 0.5 * inputs['rho'] * inputs['u']**2 * inputs['CL']
        else:
            partials['lift', 'rho'] = dq_drho * inputs['CL'] * inputs['s_ref'] * inputs['eta']
            partials['lift', 'u'] = dq_dV * inputs['CL'] * inputs['s_ref'] * inputs['eta']
            partials['lift', 'CL'] = 0.5 * inputs['rho'] * inputs['u']**2 * inputs['s_ref'] * inputs['eta']
            partials['lift', 's_ref'] = 0.5 * inputs['rho'] * inputs['u']**2 * inputs['CL'] * inputs['eta']
            partials['lift', 'eta'] = 0.5 * inputs['rho'] * inputs['u']**2 * inputs['CL'] * inputs['s_ref']


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Test both wing and canard
    for surface in ['wing', 'canard']:
        print(f'\nTesting {surface.upper()} configuration:')
        
        # Create new model for each test
        prob.model = om.Group()
        
        # Create IndepVarComp for inputs
        ivc = om.IndepVarComp()
        
        # Flow conditions
        ivc.add_output('rho', val=1.225, units='kg/m**3')
        ivc.add_output('u', val=100.0, units='m/s')
        
        # Surface parameters
        if surface == 'wing':
            ivc.add_output('CL', val=0.5)
            ivc.add_output('s_ref', val=120.0, units='m**2')
        else:
            ivc.add_output('CL', val=0.3)
            ivc.add_output('s_ref', val=20.0, units='m**2')
            ivc.add_output('eta', val=0.95)
        
        # Add IVC and lift force component to model
        prob.model.add_subsystem('inputs', ivc, promotes=['*'])
        prob.model.add_subsystem('lift', 
                                LiftForce(surface_type=surface),
                                promotes=['*'])
        
        # Setup and run problem
        prob.setup()
        prob.run_model()
        
        # Print results
        print('\nFlow Conditions:')
        print(f'  Density:    {prob.get_val("rho")} kg/m^3')
        print(f'  Velocity:   {prob.get_val("u")} m/s')
        
        print(f'\n{surface.capitalize()}:')
        print(f'  CL:         {prob.get_val("CL")}')
        print(f'  Area:       {prob.get_val("s_ref")} m^2')
        if surface == 'canard':
            print(f'  eta:        {prob.get_val("eta")}')
        print(f'  Lift:       {prob.get_val("lift")} N')
        
        # Check partials
        print('\nChecking partials...')
        prob.check_partials(compact_print=True) 