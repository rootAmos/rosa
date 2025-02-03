import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt

class WaveDrag(om.ExplicitComponent):
    """
    Calculates the wave drag increment using equation 13.25.
    
    Inputs:
        M : float
            Mach number [-]
        M_crit : float
            Critical Mach number [-]
        a : float
            Wave drag coefficient a [-]
        b : float
            Wave drag coefficient b [-]
    
    Outputs:
        CD_wave : float
            Wave drag coefficient increment [-]
    """

    def initialize(self):
        self.options.declare('M_crit', default=0.88, desc='Critical Mach number')
        self.options.declare('a', default=0.1498, desc='Wave drag coefficient a')
        self.options.declare('b', default=3.20, desc='Wave drag coefficient b')
        self.options.declare('N', default=1, desc='Number of nodes')
    
    def setup(self):

        N = self.options['N']

        # Inputs
        self.add_input('mach', val=1.0 * np.ones(N), desc='Mach number')
        
        # Outputs
        self.add_output('CD_wave', val=1.0 * np.ones(N), desc='Wave drag coefficient increment')

        

        # Declare partials
        self.declare_partials('CD_wave', ['mach'])
        
    def compute(self, inputs, outputs):

        # Unpack options
        M_crit = self.options['M_crit']
        a = self.options['a']
        b = self.options['b']

        # Unpack inputs
        M = inputs['mach']
        
        # Only compute wave drag if M > M_crit
        outputs['CD_wave'] = np.where(M > M_crit, a * ((M/M_crit) - 1)**b, 0.0)
        
    def compute_partials(self, inputs, partials):

        N = self.options['N']

        # Unpack options
        M_crit = self.options['M_crit']
        a = self.options['a']
        b = self.options['b']

        # Unpack inputs
        M = inputs['mach']
    
        partials['CD_wave', 'mach'] = np.where(M > M_crit, np.eye(N) * (a * b * ((M/M_crit) - 1)**(b-1)) / M_crit, np.eye(N) * 0.0)



if __name__ == "__main__":
    
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('mach', val=0.5, desc='Mach number')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('wave_drag', WaveDrag(M_crit=0.85, a=0.1498, b=3.2), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Create Mach sweep
    M_range = np.linspace(0.5, 1.0, 100)
    CD_wave = []
    
    for M in M_range:
        prob.set_val('mach', M)
        prob.run_model()
        CD_wave.append(prob.get_val('CD_wave')[0])
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(M_range, CD_wave, 'b-', label='Wave Drag')
    plt.axvline(x=0.85, color='r', linestyle='--', label='M_crit')
    plt.grid(True)
    plt.xlabel('Mach Number')
    plt.ylabel('Wave Drag Coefficient')
    plt.title('Wave Drag vs Mach Number')
    plt.show()

    prob.check_partials(compact_print=True)