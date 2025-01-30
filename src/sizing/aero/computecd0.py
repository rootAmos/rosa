import openmdao.api as om
import numpy as np

class ZeroLiftDragComponent(om.ExplicitComponent):
    """
    Calculates zero-lift drag coefficient for a single lifting surface.
    
    Inputs:
        Cf : float
            Skin friction coefficient [-]
        FF : float
            Form factor [-]
        Q : float
            Interference factor [-]
        S_wet : float
            Wetted area [m^2]
        S_ref : float
            Reference area [m^2]
    
    Outputs:
        CD0 : float
            Zero-lift drag coefficient [-]
    """
    
    def setup(self):
        self.add_input('Cf', val=0.0, desc='Skin friction coefficient')
        self.add_input('FF', val=1.0, desc='Form factor')
        self.add_input('Q', val=1.0, desc='Interference factor')
        self.add_input('S_wet', val=0.0, units='m**2', desc='Wetted area')
        self.add_input('S_ref', val=1.0, units='m**2', desc='Reference area')
        
        self.add_output('CD0', val=0.0, desc='Zero-lift drag coefficient')
        
        self.declare_partials('CD0', ['Cf', 'FF', 'Q', 'S_wet', 'S_ref'])
        
    def compute(self, inputs, outputs):
        Cf = inputs['Cf']
        FF = inputs['FF']
        Q = inputs['Q']
        S_wet = inputs['S_wet']
        S_ref = inputs['S_ref']
        
        outputs['CD0'] = Cf * FF * Q * S_wet/S_ref
        
    def compute_partials(self, inputs, partials):
        Cf = inputs['Cf']
        FF = inputs['FF']
        Q = inputs['Q']
        S_wet = inputs['S_wet']
        S_ref = inputs['S_ref']
        
        partials['CD0', 'Cf'] = FF * Q * S_wet/S_ref
        partials['CD0', 'FF'] = Cf * Q * S_wet/S_ref
        partials['CD0', 'Q'] = Cf * FF * S_wet/S_ref
        partials['CD0', 'S_wet'] = Cf * FF * Q/S_ref
        partials['CD0', 'S_ref'] = -Cf * FF * Q * S_wet/(S_ref**2) 
        
if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp for inputs
    ivc = om.IndepVarComp()
    ivc.add_output('Cf', val=0.003, desc='Skin friction coefficient')
    ivc.add_output('FF', val=1.2, desc='Form factor')
    ivc.add_output('Q', val=1.1, desc='Interference factor')
    ivc.add_output('S_wet', val=100.0, units='m**2', desc='Wetted area')
    ivc.add_output('S_ref', val=50.0, units='m**2', desc='Reference area')
    
    # Add IVC and CD0 component to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cd0', ZeroLiftDragComponent(), promotes=['*'])
    
    # Setup and run problem
    prob.setup()
    prob.run_model()
    
    # Print results
    print('Inputs:')
    print(f'  Cf = {prob.get_val("Cf")}')
    print(f'  FF = {prob.get_val("FF")}')
    print(f'  Q = {prob.get_val("Q")}')
    print(f'  S_wet = {prob.get_val("S_wet")} m^2')
    print(f'  S_ref = {prob.get_val("S_ref")} m^2')
    print('\nOutput:')
    print(f'  CD0 = {prob.get_val("CD0")}')
    
    # Check partials
    prob.check_partials(compact_print=True)