import openmdao.api as om
import numpy as np

class MachNumber(om.ExplicitComponent):
    """
    Computes Mach number from velocity and speed of sound.
    M = V/a, where a = sqrt(gamma * R * T)
    """

    def initialize(self):
        self.options.declare('gamma', default=1.4,
                      desc='Ratio of specific heats')
        self.options.declare('R', default=287.05,
                      desc='Gas constant')
        self.options.declare('N', default=1,
                      desc='Number of nodes')
        

    def setup(self):
        # Inputs

        N = self.options['N']

        self.add_input('u', val=1.0 * np.ones(N), units='m/s',
                      desc='Flow velocity')
        self.add_input('temp', val=1.0 * np.ones(N), units='K',
                      desc='Static temperature')



        
        # Outputs
        self.add_output('mach', val=1.0 * np.ones(N),
                       desc='Mach number')
        

        # Declare partials
        self.declare_partials('mach', ['u', 'temp'])
        
    def compute(self, inputs, outputs):

        # Unpack inputs
        V = inputs['u'] # Velocity (m/s)    
        T = inputs['temp'] # Temperature (K)

        # Unpack constants
        gamma = self.options['gamma']
        R = self.options['R']

        # Speed of sound
        a = np.sqrt(gamma * R * T)
        
        # Mach number
        outputs['mach'] = V / a

        
    def compute_partials(self, inputs, partials):

        # Unpack inputs
        V = inputs['u']
        T = inputs['temp']

        N = self.options['N']

        # Unpack constants
        gamma = self.options['gamma']
        R = self.options['R']

        # Speed of sound
        a = np.sqrt(gamma * R * T)

        
        partials['mach', 'u'] = np.eye(N) * 1 / a
        partials['mach', 'temp'] = np.eye(N) * -V / (2 * a * T)


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('u', val=100.0, units='m/s')
    ivc.add_output('temp', val=288.15, units='K')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('mach', MachNumber(), promotes=['*'])
    
    # Setup and run problem
    prob.setup()
    prob.run_model()
    
    # Print results
    print('\nResults:')
    print(f'Velocity:     {prob.get_val("u")[0]:.1f} m/s')
    print(f'Temperature:  {prob.get_val("temp")[0]:.1f} K')
    print(f'Mach number: {prob.get_val("mach")[0]:.3f}')
    

    # Run a quick parameter study
    import matplotlib.pyplot as plt
    
    velocities = np.linspace(0, 400, 100)
    mach_numbers = []
    
    for v in velocities:
        prob.set_val('u', v)
        prob.run_model()
        mach_numbers.append(prob.get_val('mach')[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(velocities, mach_numbers)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Mach Number')
    plt.grid(True)
    plt.title('Mach Number vs. Velocity at Sea Level')
    plt.show() 

    prob.check_partials(compact_print=True)