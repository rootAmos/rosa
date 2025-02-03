import openmdao.api as om
import numpy as np

class PodWettedArea(om.ExplicitComponent):
    """
    Calculates the wetted area of a pod/plug using equation 13.14.
    
    Inputs:
        l_pod : float
            Pod length [m]
        d_pod : float
            Pod diameter [m]
    
    Outputs:
        S_wet_pod : float
            Pod wetted area [m^2]
    """
    
    def setup(self):
        # Inputs
        self.add_input('l_pod', val=1.0, units='m', desc='Pod length')
        self.add_input('d_pod', val=1.0, units='m', desc='Pod diameter')
        

        # Output
        self.add_output('S_wet_pod', val=1.0, units='m**2', desc='Pod wetted area')
        

        # Declare partials
        self.declare_partials('S_wet_pod', ['l_pod', 'd_pod'])
        
    def compute(self, inputs, outputs):
        l_pod = inputs['l_pod']
        d_pod = inputs['d_pod']
        
        # Compute wetted area using equation 13.14
        outputs['S_wet_pod'] = 0.7 * np.pi * l_pod * d_pod
        
        
    def compute_partials(self, inputs, partials):
        l_pod = inputs['l_pod']
        d_pod = inputs['d_pod']
        
        # Derivatives
        partials['S_wet_pod', 'l_pod'] = 0.7 * np.pi * d_pod
        partials['S_wet_pod', 'd_pod'] = 0.7 * np.pi * l_pod 

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
    prob.model.add_subsystem('pod', PodWettedArea(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  Pod Length:           {prob.get_val("l_pod")[0]:8.3f} m')
    print(f'  Pod Diameter:         {prob.get_val("d_pod")[0]:8.3f} m')
    print(f'  Wetted Area:         {prob.get_val("S_wet_pod")[0]:8.3f} m²')
    
    # Parameter sweeps
    print('\nParameter Sweeps:')
    print('----------------')
    
    # Create figure with 2 subplots
    plt.figure(figsize=(12, 5))
    
    # Length sweep
    l_range = np.linspace(2.0, 8.0, 50)
    S_wet_l = []
    for l in l_range:
        prob.set_val('l_pod', l)
        prob.run_model()
        S_wet_l.append(prob.get_val('S_wet_pod')[0])
    
    plt.subplot(121)
    plt.plot(l_range, S_wet_l)
    plt.xlabel('Pod Length (m)')
    plt.ylabel('Wetted Area (m²)')
    plt.grid(True)
    plt.title('Effect of Pod Length')
    
    # Diameter sweep
    prob.set_val('l_pod', 4.0)  # Reset to baseline
    d_range = np.linspace(1.0, 4.0, 50)
    S_wet_d = []
    for d in d_range:
        prob.set_val('d_pod', d)
        prob.run_model()
        S_wet_d.append(prob.get_val('S_wet_pod')[0])
    
    plt.subplot(122)
    plt.plot(d_range, S_wet_d)
    plt.xlabel('Pod Diameter (m)')
    plt.ylabel('Wetted Area (m²)')
    plt.grid(True)
    plt.title('Effect of Pod Diameter')
    
    plt.tight_layout()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.show() 