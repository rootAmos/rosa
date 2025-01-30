import openmdao.api as om
import numpy as np

class WingLift(om.ExplicitComponent):
    """
    Calculates lift coefficient for wing.
    
    Inputs:
        CL_alpha_w : float
            Wing lift curve slope [1/rad]
        alpha : float
            Angle of attack [rad]
        i_w : float
            Wing incidence angle [rad]
        alpha_0_w : float
            Wing zero-lift angle of attack [rad]
        epsilon : float
            Downwash angle [rad]
    
    Outputs:
        CL_w : float
            Wing lift coefficient [-]
    """
    
    def setup(self):
        # Wing inputs
        self.add_input('CL_alpha_w', val=0.0, units='1/rad',
                      desc='Wing lift curve slope')
        self.add_input('alpha', val=0.0, units='rad',
                      desc='Angle of attack')
        self.add_input('i_w', val=0.0, units='rad',
                      desc='Wing incidence angle')
        self.add_input('alpha_0_w', val=0.0, units='rad',
                      desc='Wing zero-lift angle')
        self.add_input('epsilon', val=0.0, units='rad',
                      desc='Downwash angle')
        
        # Output
        self.add_output('CL_w', val=0.0,
                       desc='Wing lift coefficient')
        
        # Declare partials
        self.declare_partials('CL_w', ['CL_alpha_w', 'alpha', 'i_w', 
                                      'alpha_0_w', 'epsilon'])
        
    def compute(self, inputs, outputs):
        # Wing lift (with downwash)
        outputs['CL_w'] = inputs['CL_alpha_w'] * (
            inputs['alpha'] + inputs['i_w'] - inputs['epsilon'] - inputs['alpha_0_w']
        )
        
    def compute_partials(self, inputs, partials):
        # Wing derivatives
        partials['CL_w', 'CL_alpha_w'] = (
            inputs['alpha'] + inputs['i_w'] - inputs['epsilon'] - inputs['alpha_0_w']
        )
        partials['CL_w', 'alpha'] = inputs['CL_alpha_w']
        partials['CL_w', 'i_w'] = inputs['CL_alpha_w']
        partials['CL_w', 'alpha_0_w'] = -inputs['CL_alpha_w']
        partials['CL_w', 'epsilon'] = -inputs['CL_alpha_w']


class CanardLift(om.ExplicitComponent):
    """
    Calculates lift coefficient for canard.
    
    Inputs:
        CL_alpha_c : float
            Canard lift curve slope [1/rad]
        alpha : float
            Angle of attack [rad]
        i_c : float
            Canard incidence angle [rad]
        alpha_0_c : float
            Canard zero-lift angle of attack [rad]
    
    Outputs:
        CL_c : float
            Canard lift coefficient [-]
    """
    
    def setup(self):
        # Canard inputs
        self.add_input('CL_alpha_c', val=0.0, units='1/rad',
                      desc='Canard lift curve slope')
        self.add_input('alpha', val=0.0, units='rad',
                      desc='Angle of attack')
        self.add_input('i_c', val=0.0, units='rad',
                      desc='Canard incidence angle')
        self.add_input('alpha_0_c', val=0.0, units='rad',
                      desc='Canard zero-lift angle')
        
        # Output
        self.add_output('CL_c', val=0.0,
                       desc='Canard lift coefficient')
        
        # Declare partials
        self.declare_partials('CL_c', ['CL_alpha_c', 'alpha', 'i_c', 'alpha_0_c'])
        
    def compute(self, inputs, outputs):
        # Canard lift (no downwash)
        outputs['CL_c'] = inputs['CL_alpha_c'] * (
            inputs['alpha'] + inputs['i_c'] - inputs['alpha_0_c']
        )
        
    def compute_partials(self, inputs, partials):
        # Canard derivatives
        partials['CL_c', 'CL_alpha_c'] = (
            inputs['alpha'] + inputs['i_c'] - inputs['alpha_0_c']
        )
        partials['CL_c', 'alpha'] = inputs['CL_alpha_c']
        partials['CL_c', 'i_c'] = inputs['CL_alpha_c']
        partials['CL_c', 'alpha_0_c'] = -inputs['CL_alpha_c']


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp for inputs
    ivc = om.IndepVarComp()
    
    # Wing inputs
    ivc.add_output('CL_alpha_w', val=6.0, units='1/rad')
    ivc.add_output('i_w', val=2.0, units='deg')
    ivc.add_output('alpha_0_w', val=-2.0, units='deg')
    ivc.add_output('epsilon', val=2.0, units='deg')
    
    # Canard inputs
    ivc.add_output('CL_alpha_c', val=5.0, units='1/rad')
    ivc.add_output('i_c', val=0.0, units='deg')
    ivc.add_output('alpha_0_c', val=-1.0, units='deg')
    
    # Common input
    ivc.add_output('alpha', val=4.0, units='deg')
    
    # Add IVC and components to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('wing', WingLift(), promotes=['*'])
    prob.model.add_subsystem('canard', CanardLift(), promotes=['*'])
    
    # Setup and run problem
    prob.setup()
    prob.run_model()
    
    # Print results
    print('\nInputs:')
    print(f'  Alpha:      {prob["alpha"]} deg')
    print(f'  Epsilon:    {prob["epsilon"]} deg')
    print('\nWing:')
    print(f'  CL_alpha_w: {prob["CL_alpha_w"]} 1/rad')
    print(f'  i_w:        {prob["i_w"]} deg')
    print(f'  alpha_0_w:  {prob["alpha_0_w"]} deg')
    print(f'  CL_w:       {prob["CL_w"]}')
    print('\nCanard:')
    print(f'  CL_alpha_c: {prob["CL_alpha_c"]} 1/rad')
    print(f'  i_c:        {prob["i_c"]} deg')
    print(f'  alpha_0_c:  {prob["alpha_0_c"]} deg')
    print(f'  CL_c:       {prob["CL_c"]}')
    
    # Check partials
    prob.check_partials(compact_print=True) 