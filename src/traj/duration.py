import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
import pdb
import matplotlib.pyplot as plt

class Duration:
    def __init__(self, phase):
        self.phase = phase

    def analyze(self, phase, dur_s, mode):


        dt_s = dur_s / phase['N']
        time_dt_s = np.arange(phase['t0_s'], phase['t0_s'] + dur_s, dt_s)

        if phase['mode'] == 'time':
            pass
        # end

        if phase['mode'] == 'distance':

            u_m_s = cumulative_trapezoid(phase['udot_m_s2'], time_dt_s) + phase['u0_m_s']
            u_m_s = np.insert(u_m_s, 0, phase['u0_m_s'])

            xdot_m_s = np.cos(phase['gamma_rad']) * u_m_s
            x_m = cumulative_trapezoid(xdot_m_s, time_dt_s) + phase['x0_m']
            x_m = np.insert(x_m, 0, phase['x0_m'])
            dist_m = x_m[-1] - phase['x0_m']

            error = np.abs(dist_m - phase['dist_tgt_m'])

            return error

            # end

        elif phase['mode'] == 'accel':

            u_m_s = cumulative_trapezoid(phase['udot_m_s2'], time_dt_s) + phase['u0_m_s']
            u_m_s = np.insert(u_m_s, 0, phase['u0_m_s'])
            error = np.abs(u_m_s[-1] - phase['u1_tgt_m_s'])

            return error

        elif phase['mode'] == 'alt':

            u_m_s = cumulative_trapezoid(phase['udot_m_s2'], time_dt_s) + phase['u0_m_s']
            u_m_s = np.insert(u_m_s, 0, phase['u0_m_s'])
            zdot_m_s = np.sin(phase['gamma_rad']) * u_m_s

            z_m = cumulative_trapezoid(zdot_m_s, time_dt_s) + phase['z0_m']
            z_m = np.insert(z_m, 0, phase['z0_m'])
            error = np.abs(z_m[-1] - phase['z1_tgt_m'])

            return error
        
    def solve_duration(self, phase):

        if phase['mode'] != 'time':

            def objective(dur_s):
                return self.analyze(phase, dur_s, phase['mode'])
            
            # Optimize duration (bounds prevent negative or extremely large durations)
            result = minimize_scalar(objective, bounds=(0.1, 100), method='bounded')

            phase['dur_s'] = result.x
        # end

        phase = self.complete_phase(phase)

        return phase


    def complete_phase(self, phase):

        # Recompute phase results

        phase['dt_s'] = phase['dur_s'] / phase['N']
        phase['time_s'] = np.arange(phase['t0_s'], phase['t0_s'] + phase['dur_s'], phase['dt_s'])


        u_m_s = cumulative_trapezoid(phase['udot_m_s2'], phase['time_s']) + phase['u0_m_s']
        u_m_s = np.insert(u_m_s, 0, phase['u0_m_s'])
        xdot_m_s = np.cos(phase['gamma_rad']) * u_m_s
        zdot_m_s = np.sin(phase['gamma_rad']) * u_m_s

        x_m = cumulative_trapezoid(xdot_m_s, phase['time_s']) + phase['x0_m']
        x_m = np.insert(x_m, 0, phase['x0_m'])
        z_m = cumulative_trapezoid(zdot_m_s, phase['time_s']) + phase['z0_m']
        z_m = np.insert(z_m, 0, phase['z0_m'])

        phase['x_m'] = x_m
        phase['z_m'] = z_m
        
        # Compute phase results
        phase['t1_s'] = phase['t0_s'] + phase['dur_s']
        phase['x1_m'] = phase['x_m'][-1]
        phase['z1_m'] = phase['z_m'][-1]
        phase['u1_m_s'] = u_m_s[-1]
        phase['t1_s'] = phase['t0_s'] + phase['dur_s']
        phase['u_m_s'] = u_m_s
        phase['xdot_m_s'] = xdot_m_s
        phase['zdot_m_s'] = zdot_m_s
        phase['dist_m'] = phase['x_m'][-1] - phase['x0_m']


        return phase
    # end

    def plot_trajectory(self, phase):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # X-axis quantities (first column)
        axes[0,0].plot(phase['time_s'], phase['x_m'], label='x (m)')
        axes[0,0].set_ylabel('Position (m)')
        axes[0,0].grid(True)
        axes[0,0].legend()
        
        axes[1,0].plot(phase['time_s'], phase['xdot_m_s'], label='$\dot{x}$ (m/s)')
        axes[1,0].set_ylabel('Velocity (m/s)')
        axes[1,0].grid(True)
        axes[1,0].legend()
        
        # X acceleration
        axes[2,0].plot(phase['time_s'], phase['udot_m_s2'], label='$\ddot{u}$ (m/s²)')
        axes[2,0].set_ylabel('Acceleration (m/s²)')
        axes[2,0].set_xlabel('Time (s)')
        axes[2,0].grid(True)
        axes[2,0].legend()
        
        # Y-axis quantities (middle column) - all empty
        axes[0,1].grid(True)
        axes[1,1].grid(True)
        axes[2,1].grid(True)
        axes[2,1].set_xlabel('Time (s)')
        
        # Z-axis quantities (third column)
        axes[0,2].plot(phase['time_s'], phase['z_m'], label='z (m)')
        axes[0,2].set_ylabel('Position (m)')
        axes[0,2].grid(True)
        axes[0,2].legend()
        
        axes[1,2].plot(phase['time_s'], phase['zdot_m_s'], label='$\dot{z}$ (m/s)')
        axes[1,2].set_ylabel('Velocity (m/s)')
        axes[1,2].grid(True)
        axes[1,2].legend()
        
        # No z acceleration to plot in third row
        axes[2,2].set_ylabel('Acceleration (m/s²)')
        axes[2,2].set_xlabel('Time (s)')
        axes[2,2].grid(True)
        
        plt.tight_layout(h_pad=1.0, w_pad=1.0, pad=1.5)
        
        # Create separate figure for body velocity
        fig2, ax = plt.subplots(figsize=(10, 4))
        ax.plot(phase['time_s'], phase['u_m_s'], label='u (m/s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Body Velocities')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        
        plt.show()

    
if __name__ == "__main__":

    phase = {}
    phase['N'] = 10

    # Test Options
    phase['u0_m_s'] = 50
    phase['t0_s'] = 0
    phase['x0_m'] = 0
    phase['z0_m'] = 0
    phase['dist_tgt_m'] = 1000 
    phase['udot_m_s2'] = 0.1 * np.ones(phase['N'])
    phase['gamma_rad'] = 0 * np.ones(phase['N'])
    phase['mode'] = 'distance'

    dur = Duration(phase)
    phase = dur.solve_duration(phase)

    

    #TODO: Set up acceleration and altitude test phases
    phase['dur_cond'] = 'accel'
    phase['dur_cond'] = 'alt'

    
    dur.plot_trajectory(phase)

