import numpy as np
import glob
from hvr.hover_perfo import HoverPerformance
from crz.crz_perfo import CruisePerformance
from duration import Duration
import sys
import os
import pdb
from plotting import plot_mission

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

class Mission:
    def __init__(self):
        self.hover_analyzer = HoverPerformance()
        self.cruise_analyzer = CruisePerformance({'aero_filenames': glob.glob("data/aero/*.txt")})
        self.duration = Duration({})

    def analyze(self, vehicle, hover_phase, accel_phase, cruise_phase):
        """
        Analyze a complete mission with hover and cruise segments
        """
        # First compute duration and kinematics for hover
        hover_phase = self.duration.solve_duration(hover_phase)
        
        # Then analyze hover performance
        hover_phase = self.hover_analyzer.analyze(vehicle, hover_phase)

        # Update accel initial conditions based on hover results
        accel_phase['t0_s'] = hover_phase['t1_s']
        accel_phase['x0_m'] = hover_phase['x1_m']
        accel_phase['z0_m'] = hover_phase['z1_m']
        accel_phase['u0_m_s'] = hover_phase['u1_m_s']
        
        # compute accel duration
        accel_phase = self.duration.solve_duration(accel_phase)

        # analyze accel performance
        accel_phase = self.cruise_analyzer.analyze(vehicle, accel_phase)

        # Update cruise initial conditions based on accel results
        cruise_phase['t0_s'] = accel_phase['t1_s']
        cruise_phase['x0_m'] = accel_phase['x1_m']
        cruise_phase['z0_m'] = accel_phase['z1_m']
        cruise_phase['u0_m_s'] = accel_phase['u1_m_s']
        
        # Compute duration and kinematics for cruise
        cruise_phase = self.duration.solve_duration(cruise_phase)        
        # Then analyze cruise performance
        cruise_phase = self.cruise_analyzer.analyze(vehicle, cruise_phase)
        
        return hover_phase, accel_phase, cruise_phase

if __name__ == "__main__":
    # Define vehicle dictionary
    vehicle = {
        'mtom_kg': 5500,
        'fwd_prplsr_diam_m': 1.58,
        'lift_prplsr_diam_m': 3.09,
        'num_fwd_prplsrs': 2,
        'num_lift_prplsrs': 8,
        'fwd_prplsr_rpm_min': 500,
        'fwd_prplsr_rpm_max': 4000,
        'lift_prplsr_rpm_min': 500,
        'lift_prplsr_rpm_max': 4000,
        'wingarea_m2': 23,
        'lift_prplsr_beta_min': -7,
        'lift_prplsr_beta_max': 21
    }

    # Define hover phase dictionary
    len_hover = 100
    hover_phase = {
        'N': len_hover,
        'density_kgm3': 1.05 * np.ones(len_hover),
        'udot_m_s2': np.zeros(len_hover),
        #'zdot_m_s': np.zeros(len_hover),
        'zddot_m_s2': 0.025 * np.ones(len_hover),
        'u0_m_s': 1e-3,
        'zdot0_m_s': 1e-3,
        't0_s': 0,
        'x0_m': 0,
        'z0_m': 0,
        'dur_s': 30,  # 30 seconds hover
        'gamma_rad': np.pi/2 * np.ones(len_hover),  # Point straight up
        'end_cond': 'alt',
        'ver_thrust': True,
        'name': 'Liftoff',
        'z1_tgt_m': 100 / 3.048
    }

    len_accel = 100
    accel_phase = {
        'N': len_accel,
        'density_kgm3': 1.05 * np.ones(len_accel),
        'udot_m_s2': 0.5 * np.ones(len_accel),
        'zdot_m_s': 0 * np.ones(len_accel),
        'zddot_m_s2': 0 * np.ones(len_accel),
        'gamma_rad': 0 * np.ones(len_accel),
        'end_cond': 'accel',
        'u1_tgt_m_s': 65,
        'zdot0_m_s': 1e-3,
        't0_s': 0,
        'x0_m': 0,
        'z0_m': 0,
        'ver_thrust': True,
        'power_lift_motor_W': np.linspace(250,10,len_accel) * 1000,
        'name': 'Transition'

    }

    # Define cruise phase dictionary
    len_cruise = 100
    cruise_phase = {
        'N': len_cruise,
        'density_kgm3': 1.05 * np.ones(len_cruise),
        'dist_tgt_m': 10*1000,  # Target distance to travel
        'udot_m_s2': 0 * np.ones(len_cruise),  # Acceleration
        'gamma_rad': 0 * np.ones(len_cruise),  # Level flight
        'end_cond': 'distance',
        'ver_thrust': False,
        'aero_filenames': glob.glob("data/aero/*.txt"),
        'name': 'Cruise'
    }

    # Create and run mission
    mission = Mission()
    hover_phase, accel_phase, cruise_phase = mission.analyze(vehicle, hover_phase, accel_phase, cruise_phase)

    """
    # Print results
    print("\nHover Phase Results:")
    print(f"Duration (s): {hover_phase['dur_s']}")
    print(f"Final altitude (m): {hover_phase['z1_m']}")
    print(f"Thrust per motor (N): {hover_phase['thrust_unit_N']}")
    print(f"Power per motor (kW): {hover_phase['power_lift_motor_W']/1000}")
    print(f"Total Electric Power (kW): {hover_phase['power_elec_W']/1000}")

    print("\nCruise Phase Results:")
    print(f"Duration (s): {cruise_phase['dur_s']}")
    print(f"Distance traveled (m): {cruise_phase['dist_m']}")
    print(f"Thrust per motor (N): {cruise_phase['thrust_unit_N'][0]}")
    print(f"Power per motor (kW): {cruise_phase['power_fwd_motor_W'][0]/1000}")
    print(f"Total Electric Power (kW): {cruise_phase['power_elec_W'][0]/1000}")
    """

    # Plot trajectories
    #pdb.set_trace()
    phases = [hover_phase, accel_phase, cruise_phase]

    plot_mission(phases)


