import openmdao.api as om
import numpy as np

class SkinFriction(om.ExplicitComponent):
    """
    Calculates skin friction coefficient using a weighted combination 
    of laminar and turbulent contributions.
    
    Inputs:
        Re : float
            Reynolds number [-]
        M : float
            Mach number [-]
        k_lam : float
            Laminar flow fraction [-]
    
    Outputs:
        Cf : float
            Combined skin friction coefficient [-]
    """
    
    def setup(self):
        # Inputs
        self.add_input('Re', val=1e6, desc='Reynolds number')
        self.add_input('mach', val=0.0, desc='Mach number')
        self.add_input('k_lam', val=0.0, desc='Laminar flow fraction')
        
        # Outputs
        self.add_output('Cf', val=0.0, desc='Skin friction coefficient')
        
        # Declare partials
        self.declare_partials('Cf', ['Re', 'mach', 'k_lam'])
        
    def compute(self, inputs, outputs):
        Re = inputs['Re']
        M = inputs['mach']
        k_lam = inputs['k_lam']
        
        # Calculate components
        Cf_lam = 1.328 / np.sqrt(Re)
        Cf_turb = 0.455 / (np.log10(Re)**2.58 * (1 + 0.144*M**2)**0.65)
        
        # Combine using weighted sum
        outputs['Cf'] = k_lam * Cf_lam + (1 - k_lam) * Cf_turb
        
    def compute_partials(self, inputs, partials):
        Re = inputs['Re']
        M = inputs['mach']
        k_lam = inputs['k_lam']
        
        # Combined derivatives
        partials['Cf', 'Re'] = (0.5098*(k_lam - 1))/(Re*(0.4343*np.log(Re))**3.5800*(0.1440*M**2 + 1)**0.6500) - (0.6640*k_lam)/Re**1.5000

        partials['Cf', 'mach'] =  (0.0852*M*(k_lam - 1))/((0.4343*np.log(Re))**2.5800*(0.1440*M**2 + 1)**1.6500)

        partials['Cf', 'k_lam'] = 1.3280/Re**0.5000 - 0.4550/((0.4343*np.log(Re))**2.5800*(0.1440*M**2 + 1)**0.6500)



if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp for inputs
    ivc = om.IndepVarComp()
    ivc.add_output('Re', val=1e7)
    ivc.add_output('mach', val=0.3)
    ivc.add_output('k_lam', val=0.1)
    
    # Add IVC and skin friction component to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('skin_friction', SkinFriction(), promotes=['*'])
    
    # Setup and run problem
    prob.setup()
    prob.run_model()
    
    # Print results
    print('\nInputs:')
    print(f'  Reynolds number: {prob["Re"]}')
    print(f'  Mach number:     {prob["M"]}')
    print(f'  Laminar fraction:{prob["k_lam"]}')
    print('\nOutput:')
    print(f'  Cf:             {prob["Cf"]}')
    
    # Check partials
    prob.check_partials(compact_print=True)