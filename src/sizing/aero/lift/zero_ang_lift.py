import openmdao.api as om
import numpy as np

class ZeroLiftAngle(om.ExplicitComponent):
    """
    Calculates the zero-lift angle of attack with Mach number corrections.
    
    Inputs:
        alpha0_airfoil : float
            Basic zero-lift angle of attack at M=0.3 [rad]
        delta_alpha_0_d_twist : float
            Change in zero-lift angle due to wing twist [rad]
        epsilon_t : float
            Twist angle [rad]
    
    Outputs:
        alpha0 : float
            lifting surface zero-lift angle with Mach correction [rad]
    """
    def initialize(self):
        self.options.declare('d_alpha0_d_twist', default=-0.405, types=float)
    

    def setup(self):
        # Inputs
        self.add_input('alpha0_airfoil', val=0.0, units='rad', 
                      desc='airfoil zero-lift angle')
        self.add_input('d_twist', val=0.0, units='rad', 
                      desc='twist angle')
        
        # Outputs
        self.add_output('alpha0', val=0.0, units='rad', 
                       desc='lifting surface zero-lift angle with Mach correction')
        
        # Declare partials
        self.declare_partials('alpha0', ['*'], method='exact')
        
    def compute(self, inputs, outputs):

        d_alpha0_d_twist = self.options['d_alpha0_d_twist']


        alpha0_airfoil = inputs['alpha0_airfoil']
        d_twist = inputs['d_twist']

        

        # Equation from image
        outputs['alpha0'] = (alpha0_airfoil + d_alpha0_d_twist * d_twist)
                               
        
    def compute_partials(self, inputs, partials):

        # Inputs
        d_twist = inputs['d_twist']
        alpha0_airfoil = inputs['alpha0_airfoil']

        # Options
        d_alpha0_d_twist = self.options['d_alpha0_d_twist']
        
        # Derivatives
        partials['alpha0', 'alpha0_airfoil'] = 1.0
        partials['alpha0', 'd_twist'] = d_alpha0_d_twist


class ZeroAngleLift(om.ExplicitComponent):
    """
    Calculates the zero-angle lift coefficient.
    
    Inputs:
        CL_alpha_eff : float
            Lift curve slope [1/rad]
        alpha0 : float
            Zero-lift angle of attack [rad]
    
    Outputs:
        CL0 : float
            Zero-angle lift coefficient [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('CL_alpha_eff', val=0.0, units='1/rad', 
                      desc='Lift curve slope')
        self.add_input('alpha0', val=0.0, units='rad', 
                      desc='Zero-lift angle of attack')
        
        # Outputs
        self.add_output('CL0', val=0.0, 
                       desc='Zero-angle lift coefficient')
        
        # Declare partials
        self.declare_partials('CL0', ['CL_alpha_eff', 'alpha0'])
        
    def compute(self, inputs, outputs):
        CL_alpha_eff = inputs['CL_alpha_eff']
        alpha0 = inputs['alpha0']
        
        # Equation: CL0 = -CL_alpha_eff * alpha0
        outputs['CL0'] = -CL_alpha_eff * alpha0
        
    def compute_partials(self, inputs, partials):
        CL_alpha_eff = inputs['CL_alpha_eff']
        alpha0 = inputs['alpha0']
        
        # Derivatives
        partials['CL0', 'CL_alpha_eff'] = -alpha0
        partials['CL0', 'alpha0'] = -CL_alpha_eff


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('CL_alpha_eff', val=5.5, units='1/rad', desc='Lift curve slope')
    ivc.add_output('alpha0', val=-2.0, units='deg', desc='Zero-lift angle of attack')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('CL0_comp', ZeroAngleLift(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  CL_alpha_eff:            {prob.get_val("CL_alpha_eff")[0]:8.3f} /rad')
    print(f'  Alpha_0:             {np.degrees(prob.get_val("alpha0")[0]):8.3f} deg')
    print(f'  CL_0:                {prob.get_val("CL0")[0]:8.3f}')
    
    # Parameter sweeps
    import matplotlib.pyplot as plt
    
    # Create figure with 2 subplots
    plt.figure(figsize=(12, 5))
    
    # CL_alpha_eff sweep
    CL_alpha_eff_range = np.linspace(4.0, 7.0, 50)
    CL0_values = []
    for cla in CL_alpha_eff_range:
        prob.set_val('CL_alpha_eff', cla)
        prob.run_model()
        CL0_values.append(prob.get_val('CL0')[0])
    
    plt.subplot(121)
    plt.plot(CL_alpha_eff_range, CL0_values)
    plt.xlabel('CL_alpha_eff [1/rad]')
    plt.ylabel('CL_0')
    plt.grid(True)
    plt.title('Effect of Lift Curve Slope')
    
    # Alpha0 sweep
    prob.set_val('CL_alpha_eff', 5.5)  # Reset to baseline
    alpha0_range = np.linspace(-4.0, 0.0, 50)  # degrees
    CL0_values = []
    for a0 in alpha0_range:
        prob.set_val('alpha0', np.radians(a0))
        prob.run_model()
        CL0_values.append(prob.get_val('CL0')[0])
    
    plt.subplot(122)
    plt.plot(alpha0_range, CL0_values)
    plt.xlabel('Alpha_0 [deg]')
    plt.ylabel('CL_0')
    plt.grid(True)
    plt.title('Effect of Zero-Lift Angle')
    
    plt.tight_layout()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.show()