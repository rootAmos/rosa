import openmdao.api as om
import numpy as np

class InducedDrag(om.ExplicitComponent):
    """
    Calculates induced drag coefficient for a single lifting surface.
    
    Inputs:
        CL : float
            Lift coefficient [-]
        aspect_ratio : float
            Aspect ratio [-]
        e : float
            Oswald efficiency factor [-]
    
    Outputs:
        CDi : float
            Induced drag coefficient [-]
    """

    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')
    
    def setup(self):

        N = self.options['N']
        # Inputs

        self.add_input('CL', val=1.0 * np.ones(N), desc='Lift coefficient')
        self.add_input('aspect_ratio', val=1.0 , desc='Aspect ratio')
        self.add_input('oswald_no', val=1.0 , desc='Oswald efficiency factor')
        self.add_input('CL0', val=1.0 * np.ones(N), desc='Reference area')
        


        # Output
        self.add_output('CDi', val=1.0 * np.ones(N), desc='Induced drag coefficient')
        

        # Declare partials
        self.declare_partials('CDi', ['*'])
        
    def compute(self, inputs, outputs):

        CL = inputs['CL']
        aspect_ratio = inputs['aspect_ratio']
        oswald_no = inputs['oswald_no']
        CL0 = inputs['CL0']

        # Compute induced drag coefficient
        outputs['CDi'] = (CL- CL0)**2 / (np.pi * aspect_ratio *oswald_no)
        
    def compute_partials(self, inputs, partials):

        N = self.options['N']
        
        CL = inputs['CL']
        aspect_ratio = inputs['aspect_ratio']
        oswald_no = inputs['oswald_no']
        CL0 = inputs['CL0']

        # Derivatives
        partials['CDi', 'CL'] = np.eye(N) * (0.3183*(2*CL - 2*CL0))/(aspect_ratio*oswald_no)

        partials['CDi', 'aspect_ratio'] = -(0.3183*(CL - CL0)**2)/(aspect_ratio**2*oswald_no)

        partials['CDi', 'oswald_no'] = np.eye(N) * -(0.3183*(CL - CL0)**2)/(aspect_ratio*oswald_no**2)

        partials['CDi', 'CL0'] = np.eye(N) * -(0.3183*(2*CL - 2*CL0))/(aspect_ratio*oswald_no)
# end

if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('CL', val=0.5, desc='Lift coefficient')
    ivc.add_output('aspect_ratio', val=8.0, desc='Aspect ratio')
    ivc.add_output('oswald_no', val=0.85, desc='Oswald efficiency factor')
    ivc.add_output('CL0', val=0.1, desc='Zero-lift coefficient')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('CDi', InducedDrag(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  CL:                  {prob.get_val("CL")[0]:8.3f}')
    print(f'  CL0:                 {prob.get_val("CL0")[0]:8.3f}')
    print(f'  Aspect Ratio:        {prob.get_val("aspect_ratio")[0]:8.3f}')
    print(f'  Oswald Factor:       {prob.get_val("oswald_no")[0]:8.3f}')
    print(f'  CDi:                 {prob.get_val("CDi")[0]:8.6f}')
    
    # Parameter sweeps
    import matplotlib.pyplot as plt
    
    # Create figure with 2x2 subplots
    plt.figure(figsize=(12, 10))
    
    # CL sweep
    CL_range = np.linspace(0.0, 1.0, 50)
    CDi_vals = []
    for CL in CL_range:
        prob.set_val('CL', CL)
        prob.run_model()
        CDi_vals.append(prob.get_val('CDi')[0])
    
    plt.subplot(221)
    plt.plot(CL_range, CDi_vals)
    plt.xlabel('CL')
    plt.ylabel('CDi')
    plt.grid(True)
    plt.title('Effect of Lift Coefficient')
    
    # Aspect ratio sweep
    prob.set_val('CL', 0.5)  # Reset to baseline
    AR_range = np.linspace(4.0, 12.0, 50)
    CDi_vals = []
    for AR in AR_range:
        prob.set_val('aspect_ratio', AR)
        prob.run_model()
        CDi_vals.append(prob.get_val('CDi')[0])
    
    plt.subplot(222)
    plt.plot(AR_range, CDi_vals)
    plt.xlabel('Aspect Ratio')
    plt.ylabel('CDi')
    plt.grid(True)
    plt.title('Effect of Aspect Ratio')
    
    # Oswald efficiency sweep
    prob.set_val('aspect_ratio', 8.0)  # Reset to baseline
    e_range = np.linspace(0.7, 1.0, 50)
    CDi_vals = []
    for e in e_range:
        prob.set_val('oswald_no', e)
        prob.run_model()
        CDi_vals.append(prob.get_val('CDi')[0])
    
    plt.subplot(223)
    plt.plot(e_range, CDi_vals)
    plt.xlabel('Oswald Efficiency Factor')
    plt.ylabel('CDi')
    plt.grid(True)
    plt.title('Effect of Oswald Efficiency')
    
    # CL0 sweep
    prob.set_val('oswald_no', 0.85)  # Reset to baseline
    CL0_range = np.linspace(0.0, 0.4, 50)
    CDi_vals = []
    for CL0 in CL0_range:
        prob.set_val('CL0', CL0)
        prob.run_model()
        CDi_vals.append(prob.get_val('CDi')[0])
    
    plt.subplot(224)
    plt.plot(CL0_range, CDi_vals)
    plt.xlabel('CL0')
    plt.ylabel('CDi')
    plt.grid(True)
    plt.title('Effect of Zero-Lift Coefficient')
    
    plt.tight_layout()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.show()
