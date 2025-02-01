import openmdao.api as om
import numpy as np

class NacelleFormFactor(om.ExplicitComponent):
    """
    Calculates the form factor for a nacelle using equation 13.24.
    
    Inputs:
        c_duct : float
            Nacelle length [m]
        d_duct : float
            Nacelle diameter [m]
    
    Outputs:
        ff_nacelle : float
            Nacelle form factor [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('c_duct', val=0.0, units='m', desc='Nacelle length')
        self.add_input('d_duct', val=0.0, units='m', desc='Nacelle diameter')
        
        # Output
        self.add_output('ff_nacelle', val=0.0, desc='Nacelle form factor')
        
        # Declare partials
        self.declare_partials('ff_nacelle', ['c_duct', 'd_duct'])
        
    def compute(self, inputs, outputs):
        c_duct = inputs['c_duct']
        d_duct = inputs['d_duct']
        
        # Compute fineness ratio
        fineness_ratio = c_duct / d_duct
        
        # Compute form factor using equation 13.24
        outputs['ff_nacelle'] = 1.0 + 0.35/fineness_ratio
        
    def compute_partials(self, inputs, partials):
        c_duct = inputs['c_duct']
        d_duct = inputs['d_duct']
        
        # Partial derivatives
        partials['ff_nacelle', 'c_duct'] = -0.35/(c_duct**2/d_duct)
        partials['ff_nacelle', 'd_duct'] = 0.35*c_duct/(d_duct**2) 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('c_duct', val=3.0, units='m', desc='Nacelle length')
    ivc.add_output('d_duct', val=2.0, units='m', desc='Nacelle diameter')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('nacelle_ff', NacelleFormFactor(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  Nacelle Length:       {prob.get_val("c_duct")[0]:8.3f} m')
    print(f'  Nacelle Diameter:     {prob.get_val("d_duct")[0]:8.3f} m')
    print(f'  Fineness Ratio:       {prob.get_val("c_duct")[0]/prob.get_val("d_duct")[0]:8.3f}')
    print(f'  Form Factor:          {prob.get_val("ff_nacelle")[0]:8.3f}')
    
    # Parameter sweeps
    print('\nParameter Sweeps:')
    print('----------------')
    
    # Create figure with 2 subplots
    plt.figure(figsize=(12, 5))
    
    # Length sweep
    c_range = np.linspace(2.0, 6.0, 50)
    FF_c = []
    fineness_c = []
    for c in c_range:
        prob.set_val('c_duct', c)
        prob.run_model()
        FF_c.append(prob.get_val('ff_nacelle')[0])
        fineness_c.append(c/prob.get_val('d_duct')[0])
    
    plt.subplot(121)
    plt.plot(c_range, FF_c)
    plt.xlabel('Nacelle Length (m)')
    plt.ylabel('Form Factor')
    plt.grid(True)
    plt.title('Effect of Nacelle Length')
    
    # Diameter sweep
    prob.set_val('c_duct', 3.0)  # Reset to baseline
    d_range = np.linspace(1.0, 3.0, 50)
    FF_d = []
    fineness_d = []
    for d in d_range:
        prob.set_val('d_duct', d)
        prob.run_model()
        FF_d.append(prob.get_val('ff_nacelle')[0])
        fineness_d.append(prob.get_val('c_duct')[0]/d)
    
    plt.subplot(122)
    plt.plot(d_range, FF_d)
    plt.xlabel('Nacelle Diameter (m)')
    plt.ylabel('Form Factor')
    plt.grid(True)
    plt.title('Effect of Nacelle Diameter')
    
    # Additional plot showing form factor vs fineness ratio
    plt.figure(figsize=(8, 6))
    plt.plot(fineness_c, FF_c, label='Length Variation')
    plt.plot(fineness_d, FF_d, '--', label='Diameter Variation')
    plt.xlabel('Fineness Ratio (L/D)')
    plt.ylabel('Form Factor')
    plt.grid(True)
    plt.title('Form Factor vs Fineness Ratio')
    plt.legend()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.show()