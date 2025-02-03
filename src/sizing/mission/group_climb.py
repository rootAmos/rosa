import numpy as np
import openmdao.api as om
from scipy.integrate import cumulative_trapezoid
import pdb
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.sizing.mission.atmos import ComputeAtmos

class ClimbVectors(om.ExplicitComponent):
    """Computes true airspeed and vertical speed vectors"""
    
    def initialize(self):
        self.options.declare('N', types=int)
        
    def setup(self):
        N = self.options['N']
        
        # Inputs
        self.add_input('u_eas', val=80.0, units='m/s')
        self.add_input('gamma', val=5.0 * np.ones(N), units='rad')
        self.add_input('rho', val=1.225*np.ones(N), units='kg/m**3')
        
        # Outputs
        self.add_output('u', val=np.ones(N), units='m/s')
        self.add_output('zdot', val=np.ones(N), units='m/s')
        
        self.declare_partials('u', ['u_eas', 'rho'])
        self.declare_partials('zdot', ['u_eas', 'gamma', 'rho'])

    def compute(self, inputs, outputs):
        u_eas = inputs['u_eas']
        gamma = inputs['gamma']
        rho = inputs['rho']
        rho0 = 1.225
        
        # Convert EAS to TAS
        u_tas = u_eas * np.sqrt(rho0/rho)
        outputs['u'] = u_tas
        
        # Compute vertical speed
        outputs['zdot'] = u_tas * np.sin(gamma)

    def compute_partials(self, inputs, partials):
        N = self.options['N']
        u_eas = inputs['u_eas']
        gamma = inputs['gamma']
        rho = inputs['rho']
        rho0 = 1.225
        
        partials['u', 'u_eas'] = np.sqrt(rho0/rho)
        partials['u', 'rho'] = np.eye(N) * -0.5 * u_eas * np.sqrt(rho0)/rho**1.5
        
        partials['zdot', 'u_eas'] = np.sin(gamma) * np.sqrt(rho0/rho)
        partials['zdot', 'gamma'] = np.eye(N) * u_eas * np.sqrt(rho0/rho) * np.cos(gamma)
        partials['zdot', 'rho'] = np.eye(N) * -0.5 * u_eas * np.sqrt(rho0)/rho**1.5 * np.sin(gamma)

class ClimbDuration(om.ExplicitComponent):
    """Calculates final altitude given duration"""
    
    def initialize(self):
        self.options.declare('N', types=int)
        
    def setup(self):
        N = self.options['N']
        
        # Inputs
        self.add_input('zdot', val=np.ones(N), units='m/s')
        self.add_input('duration', val=1, units='s')
        self.add_input('z', val=np.ones(N), units='m')
        
        # Outputs
        self.add_output('z1_calc', val=1.0, units='m')
        self.add_output('z1_margin', val=1.0, units='m')
        
        self.declare_partials('*', '*')
        
    def compute(self, inputs, outputs):
        zdot = inputs['zdot']
        duration = inputs['duration']
        z = inputs['z']
        N = self.options['N']
        
        # Create time array
        time = np.linspace(0, duration, N).flatten()
        
        # Integrate vertical speed to get altitude
        z_calc = cumulative_trapezoid(zdot, time, initial=0)
        outputs['z1_calc'] = z_calc[-1]
        outputs['z1_margin'] = abs(z_calc[-1] - z[-1])  # Compare with target from z vector

    def compute_partials(self, inputs, partials):
        N = self.options['N']
        zdot = inputs['zdot']
        duration = inputs['duration']
        z = inputs['z']
        
        time = np.linspace(0, duration, N).flatten()
        dt = duration/(N-1)
        
        # Partial of z1_calc wrt zdot
        dz_dzdot = np.zeros(N)
        for i in range(N):
            dz_dzdot[i] = dt * (N-i)/N
        partials['z1_calc', 'zdot'] = dz_dzdot
        
        # Partial of z1_calc wrt duration
        z_calc = cumulative_trapezoid(zdot, time, initial=0)
        partials['z1_calc', 'duration'] = z_calc[-1]/duration
        
        # Partials of z1_margin
        if z_calc[-1] > z[-1]:
            sign = 1.0
        else:
            sign = -1.0
        partials['z1_margin', 'zdot'] = sign * dz_dzdot
        partials['z1_margin', 'duration'] = sign * z_calc[-1]/duration
        partials['z1_margin', 'z'] = np.zeros(N)
        partials['z1_margin', 'z'][-1] = -sign

class ClimbProfile(om.ExplicitComponent):
    """Computes time array and acceleration"""
    
    def initialize(self):
        self.options.declare('N', types=int)
        
    def setup(self):
        N = self.options['N']
        
        # Inputs
        self.add_input('duration', val=1.0, units='s')
        self.add_input('u', val=np.ones(N), units='m/s')
        self.add_input('zdot', val=np.ones(N), units='m/s')
        self.add_input('rho', val=1*np.ones(N), units='kg/m**3')
        
        # Outputs
        self.add_output('time', val=np.zeros(N), units='s')
        self.add_output('udot', val=np.zeros(N), units='m/s**2')
        
        self.declare_partials('time', 'duration')
        self.declare_partials('udot', ['duration', 'u'])
        
    def compute(self, inputs, outputs):
        N = self.options['N']
        duration = inputs['duration']
        u = inputs['u']
        
        # Create time array
        time = np.linspace(0, duration, N).flatten()
        outputs['time'] = time
        
        # Compute acceleration (N-1 points plus zero)
        du_dt = np.diff(u) / np.diff(time)
        outputs['udot'] = np.append(du_dt, 0.0)
        
    def compute_partials(self, inputs, partials):
        N = self.options['N']
        duration = inputs['duration']
        dt = float(duration/(N-1))
        
        # Partial of time wrt duration
        partials['time', 'duration'] = np.linspace(0, 1, N).flatten()
        
        # Partial of udot wrt duration
        dudot_ddur = -np.diff(inputs['u']) / (dt**2)
        partials['udot', 'duration'] = np.append(dudot_ddur, 0.0)
        
        # Partial of udot wrt u using diagonals
        partials['udot', 'u'] = np.zeros((N, N))
        diag_main = np.zeros(N)
        diag_main[:-1] = -1/dt
        diag_upper = np.zeros(N-1)
        diag_upper[:-1] = 1/dt
        
        partials['udot', 'u'] += np.diag(diag_main, k=0)
        partials['udot', 'u'] += np.diag(diag_upper, k=1)

class EndPoints(om.ExplicitComponent):
    """Generates altitude array from initial and final values"""
    
    def initialize(self):
        self.options.declare('N', desc='Number of nodes')
        
    def setup(self):
        N = self.options['N']
        
        # Inputs
        self.add_input('z0', val=1.0, units='m', desc='Initial altitude')
        self.add_input('z1', val=1.0, units='m', desc='Final altitude')
        
        # Outputs
        self.add_output('z', val=np.ones(N), units='m', desc='Altitude array')
        
        # Declare partials
        self.declare_partials('z', ['z0', 'z1'])
        
    def compute(self, inputs, outputs):
        N = self.options['N']
        z0 = inputs['z0']
        z1 = inputs['z1']
        
        outputs['z'] = np.linspace(z0, z1, N)
        
    def compute_partials(self, inputs, partials):
        N = self.options['N']
        
        # Partial of z wrt z0 (decreases linearly from 1 to 0)
        partials['z', 'z0'] = np.linspace(1, 0, N)
        
        # Partial of z wrt z1 (increases linearly from 0 to 1) 
        partials['z', 'z1'] = np.linspace(0, 1, N)

class ClimbGroup(om.Group):
    """Group for computing climb phase"""
    
    def initialize(self):
        self.options.declare('N', default=50, desc='Number of nodes')
        
    def setup(self):
        N = self.options['N']

        # Generate altitude profile first
        self.add_subsystem('end_points',
                          EndPoints(N=N),
                          promotes_inputs=['z0', 'z1'],
                          promotes_outputs=['z'])

        # Add atmospheric model
        self.add_subsystem('atmos', ComputeAtmos(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
        
        # Add climb vectors computation
        self.add_subsystem('vectors',
                          ClimbVectors(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
        
        # Add climb profile computation
        self.add_subsystem('profile',
                          ClimbProfile(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])
        
        # Add duration calculation
        self.add_subsystem('duration_calc',
                          ClimbDuration(N=N),
                          promotes_inputs=['*'],
                          promotes_outputs=['*'])

        # Connect z to altitude input for atmosphere
        self.connect('z', 'alt')

if __name__ == "__main__":
    # Create problem
    prob = om.Problem()
    
    # Create independent variables
    ivc = om.IndepVarComp()
    
    N = 50  # number of nodes
    
    # Add inputs to ivc
    ivc.add_output('u_eas', val=100.0, units='m/s')
    ivc.add_output('gamma', val=4.0 * np.ones(N), units='deg')
    ivc.add_output('duration', val=600.0, units='s')
    #ivc.add_output('rho', val=np.linspace(1.225, 0.4135, N), units='kg/m**3')
    ivc.add_output('z0', val=1500.0, units='ft')
    ivc.add_output('z1', val=30000.0, units='ft')
    




    # Add IVC and climb group to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('climb', ClimbGroup(N=N), promotes=['*'])

    
    # Setup optimization
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'IPOPT'
    
    # Define design variable
    prob.model.add_design_var('duration', lower=1e-3, upper=2000.0, units='s')
    
    # Define objective (minimize altitude margin)
    prob.model.add_objective('z1_margin')

    
    # Setup recorder
    #recorder = om.SqliteRecorder('climb_cases.sql')
    #prob.driver.add_recorder(recorder)
    #prob.driver.recording_options['includes'] = ['*']
    #prob.driver.recording_options['record_objectives'] = True
    #prob.driver.recording_options['record_desvars'] = True
    

    # Setup problem
    prob.setup()
    
    # Run optimization
    prob.run_driver()
    
    # Print results
    print(f"Optimized duration: {prob.get_val('duration')[0]:.1f} s")
    print(f"Final altitude: {prob.get_val('z1_calc')[0]:.1f} m")
    print(f"Altitude margin: {prob.get_val('z1_margin')[0]:.1f} m")
    
    # Plot results
    import matplotlib.pyplot as plt
    
    time = prob.get_val('time')
    z = prob.get_val('z') * 3.28084  # Convert to feet
    u = prob.get_val('u')
    zdot = prob.get_val('zdot') * 196.85  # Convert to ft/min
    udot = prob.get_val('udot')



    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    ax1.plot(z, u)
    ax1.set_xlabel('Altitude (ft)')
    ax1.set_ylabel('True Airspeed (m/s)')
    ax1.grid(True)

    ax2.plot(z, udot)
    ax2.set_xlabel('Altitude (ft)')
    ax2.set_ylabel('Acceleration (m/s²)') 
    ax2.grid(True)

    ax3.plot(z, zdot)
    ax3.set_xlabel('Altitude (ft)')
    ax3.set_ylabel('Climb Rate (ft/min)')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
    
    # Check partials
    #prob.check_partials(compact_print=True)