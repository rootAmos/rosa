import openmdao.api as om
import numpy as np

class PodFormFactor(om.ExplicitComponent):
    """
    Calculates the form factor for a pod/fuselage using equation 13.23.
    
    Inputs:
        l_pod : float
            Pod length [m]
        d_pod : float
            Pod diameter [m]
    
    Outputs:
        ff_pod : float
            Pod form factor [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('l_pod', val=0.0, units='m', desc='Pod length')
        self.add_input('d_pod', val=0.0, units='m', desc='Pod diameter')
        
        # Output
        self.add_output('ff_pod', val=0.0, desc='Pod form factor')
        
        # Declare partials
        self.declare_partials('ff_pod', ['l_pod', 'd_pod'])
        
    def compute(self, inputs, outputs):
        l_pod = inputs['l_pod']
        d_pod = inputs['d_pod']
        
        # Compute fineness ratio
        fineness_ratio = l_pod / d_pod
        
        # Compute form factor using equation 13.23
        outputs['ff_pod'] = (1.0 + 
                          60.0/(fineness_ratio**3) + 
                          fineness_ratio/400.0)
        
    def compute_partials(self, inputs, partials):
        l_pod = inputs['l_pod']
        d_pod = inputs['d_pod']
        
        # Partial derivatives
        # With respect to l_pod
        partials['ff_pod', 'l_pod'] = 0.0025/d_pod - (180*d_pod**3)/l_pod**4
        # With respect to d_pod
        partials['ff_pod', 'd_pod'] = (180*d_pod**2)/l_pod**3 - (0.0025*l_pod)/d_pod**2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('l_pod', val=4.0, units='m', desc='Pod length')
    ivc.add_output('d_pod', val=2.0, units='m', desc='Pod diameter')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('pod_ff', PodFormFactor(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  Pod Length:           {prob.get_val("l_pod")} m')
    print(f'  Pod Diameter:         {prob.get_val("d_pod")} m')
    print(f'  Fineness Ratio:       {prob.get_val("l_pod")/prob.get_val("d_pod")}')

    print(f'  Form Factor:          {prob.get_val("ff_pod")}')
    
    # Parameter sweeps
    print('\nParameter Sweeps:')
    print('----------------')
    
    # Create figure with 2 subplots

    plt.figure(figsize=(12, 5))
    
    # Length sweep
    l_range = np.linspace(2.0, 8.0, 50)
    FF_l = []
    fineness_l = []
    for l in l_range:
        prob.set_val('l_pod', l)
        prob.run_model()
        FF_l.append(prob.get_val('ff_pod')[0])
        fineness_l.append(l/prob.get_val('d_pod')[0])
    
    plt.subplot(121)
    plt.plot(l_range, FF_l)
    plt.xlabel('Pod Length (m)')
    plt.ylabel('Form Factor')
    plt.grid(True)
    plt.title('Effect of Pod Length')
    
    # Diameter sweep
    prob.set_val('l_pod', 4.0)  # Reset to baseline
    d_range = np.linspace(1.0, 4.0, 50)
    FF_d = []
    fineness_d = []
    for d in d_range:
        prob.set_val('d_pod', d)
        prob.run_model()
        FF_d.append(prob.get_val('ff_pod')[0])
        fineness_d.append(prob.get_val('l_pod')[0]/d)
    
    plt.subplot(122)
    plt.plot(d_range, FF_d)
    plt.xlabel('Pod Diameter (m)')
    plt.ylabel('Form Factor')
    plt.grid(True)
    plt.title('Effect of Pod Diameter')
    
    # Additional plot showing form factor vs fineness ratio
    plt.figure(figsize=(8, 6))
    plt.plot(fineness_l, FF_l, label='Length Variation')
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