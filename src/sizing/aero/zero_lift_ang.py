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
    
    def setup(self):
        # Inputs
        self.add_input('alpha0_airfoil', val=0.0, units='rad', 
                      desc='airfoil zero-lift angle')
        self.add_input('d_alpha0_d_twist', val=0.0, units='rad', 
                      desc='Change in zero-lift angle with respect to wing twist')
        self.add_input('d_twist', val=0.0, units='rad', 
                      desc='twist angle')
        
        # Outputs
        self.add_output('alpha0', val=0.0, units='rad', 
                       desc='lifting surface zero-lift angle with Mach correction')
        
        # Declare partials
        self.declare_partials('alpha0', ['alpha0_airfoil', 'd_alpha0_d_twist', 
                                          'd_twist'])
        
    def compute(self, inputs, outputs):


        alpha0_airfoil = inputs['alpha0_airfoil']
        d_alpha0_d_twist = inputs['d_alpha0_d_twist']
        d_twist = inputs['d_twist']

        
        # Equation from image
        outputs['alpha0'] = (alpha0_airfoil + d_alpha0_d_twist * d_twist)
                               
        
    def compute_partials(self, inputs, partials):

        # Inputs
        d_alpha0_d_twist = inputs['d_alpha0_d_twist']
        d_twist = inputs['d_twist']
        alpha0_airfoil = inputs['alpha0_airfoil']
        
        # Derivatives
        partials['alpha0', 'alpha0_airfoil'] = 1.0
        partials['alpha0', 'd_alpha0_d_twist'] = d_twist
        partials['alpha0', 'd_twist'] = d_alpha0_d_twist


class ZeroAngleLift(om.ExplicitComponent):
    """
    Calculates the zero-angle lift coefficient.
    
    Inputs:
        cl_alpha : float
            Lift curve slope [1/rad]
        alpha0 : float
            Zero-lift angle of attack [rad]
    
    Outputs:
        cl0 : float
            Zero-angle lift coefficient [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('cl_alpha', val=0.0, units='1/rad', 
                      desc='Lift curve slope')
        self.add_input('alpha0', val=0.0, units='rad', 
                      desc='Zero-lift angle of attack')
        
        # Outputs
        self.add_output('cl0', val=0.0, 
                       desc='Zero-angle lift coefficient')
        
        # Declare partials
        self.declare_partials('cl0', ['cl_alpha', 'alpha0'])
        
    def compute(self, inputs, outputs):
        cl_alpha = inputs['cl_alpha']
        alpha0 = inputs['alpha0']
        
        # Equation: cl0 = -cl_alpha * alpha0
        outputs['cl0'] = -cl_alpha * alpha0
        
    def compute_partials(self, inputs, partials):
        cl_alpha = inputs['cl_alpha']
        alpha0 = inputs['alpha0']
        
        # Derivatives
        partials['cl0', 'cl_alpha'] = -alpha0
        partials['cl0', 'alpha0'] = -cl_alpha


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('cl_alpha', val=5.5, units='1/rad', desc='Lift curve slope')
    ivc.add_output('alpha0', val=-2.0, units='deg', desc='Zero-lift angle of attack')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cl0_comp', ZeroAngleLift(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  CL_alpha:            {prob.get_val("cl_alpha")[0]:8.3f} /rad')
    print(f'  Alpha_0:             {np.degrees(prob.get_val("alpha0")[0]):8.3f} deg')
    print(f'  CL_0:                {prob.get_val("cl0")[0]:8.3f}')
    
    # Parameter sweeps
    import matplotlib.pyplot as plt
    
    # Create figure with 2 subplots
    plt.figure(figsize=(12, 5))
    
    # CL_alpha sweep
    cl_alpha_range = np.linspace(4.0, 7.0, 50)
    cl0_values = []
    for cla in cl_alpha_range:
        prob.set_val('cl_alpha', cla)
        prob.run_model()
        cl0_values.append(prob.get_val('cl0')[0])
    
    plt.subplot(121)
    plt.plot(cl_alpha_range, cl0_values)
    plt.xlabel('CL_alpha [1/rad]')
    plt.ylabel('CL_0')
    plt.grid(True)
    plt.title('Effect of Lift Curve Slope')
    
    # Alpha0 sweep
    prob.set_val('cl_alpha', 5.5)  # Reset to baseline
    alpha0_range = np.linspace(-4.0, 0.0, 50)  # degrees
    cl0_values = []
    for a0 in alpha0_range:
        prob.set_val('alpha0', np.radians(a0))
        prob.run_model()
        cl0_values.append(prob.get_val('cl0')[0])
    
    plt.subplot(122)
    plt.plot(alpha0_range, cl0_values)
    plt.xlabel('Alpha_0 [deg]')
    plt.ylabel('CL_0')
    plt.grid(True)
    plt.title('Effect of Zero-Lift Angle')
    
    plt.tight_layout()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.show()