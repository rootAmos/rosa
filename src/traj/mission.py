import numpy as np
import glob
from hvr.hover_perfo import HoverPerformance
from crz.crz_perfo import CruisePerformance
from duration import Duration
import sys
import os
import pdb
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

class Mission:
    def __init__(self):
        self.hover_analyzer = HoverPerformance()
        self.cruise_analyzer = CruisePerformance({'aero_filenames': glob.glob("data/aero/*.txt")})
        self.duration = Duration({})

    def analyze(self, vehicle, hover_phase, cruise_phase):
        """
        Analyze a complete mission with hover and cruise segments
        """
        # First compute duration and kinematics for hover
        hover_phase = self.duration.solve_duration(hover_phase)
        
        # Then analyze hover performance
        hover_phase = self.hover_analyzer.analyze(vehicle, hover_phase)
        
        # Update cruise initial conditions based on hover results
        cruise_phase['t0_s'] = hover_phase['t1_s']
        cruise_phase['x0_m'] = hover_phase['x1_m']
        cruise_phase['z0_m'] = hover_phase['z1_m']
        
        # Compute duration and kinematics for cruise
        cruise_phase = self.duration.solve_duration(cruise_phase)        
        # Then analyze cruise performance
        cruise_phase = self.cruise_analyzer.analyze(vehicle, cruise_phase)
        
        return hover_phase, cruise_phase

if __name__ == "__main__":
    # Define vehicle dictionary
    vehicle = {
        'mtom_kg': 5500,
        'fwd_prplsr_diam_m': 1.58,
        'lift_prplsr_diam_m': 3.09,
        'num_fwd_prplsrs': 2,
        'num_lift_prplsrs': 8,
        'fwd_prplsr_rpm_min': 2200,
        'fwd_prplsr_rpm_max': 3000,
        'lift_prplsr_rpm_min': 2200,
        'lift_prplsr_rpm_max': 3000,
        'wingarea_m2': 23,
        'lift_prplsr_beta_min': -7,
        'lift_prplsr_beta_max': 21
    }

    # Define hover phase dictionary
    len_hover = 10
    hover_phase = {
        'N': len_hover,
        'density_kgm3': 1.05 * np.ones(len_hover),
        'udot_m_s2': np.zeros(len_hover),
        'zdot_m_s': np.zeros(len_hover),
        'zddot_m_s2': 0.3,
        'u0_m_s': 1e-3,
        't0_s': 0,
        'x0_m': 0,
        'z0_m': 0,
        'dur_s': 30,  # 30 seconds hover
        'gamma_rad': np.pi/2 * np.ones(len_hover),  # Point straight up
        'mode': 'time',
        'dur_s': 60
    }

    # Define cruise phase dictionary
    len_cruise = 10
    cruise_phase = {
        'N': len_cruise,
        'density_kgm3': 1.05 * np.ones(len_cruise),
        'dist_tgt_m': 200*1000,  # Target distance to travel
        'u0_m_s': 65,
        'udot_m_s2': 0 * np.ones(len_cruise),  # Acceleration
        'gamma_rad': 0 * np.ones(len_cruise),  # Level flight
        'mode': 'distance',
        'aero_filenames': glob.glob("data/aero/*.txt")
    }

    # Create and run mission
    mission = Mission()
    hover_phase, cruise_phase = mission.analyze(vehicle, hover_phase, cruise_phase)

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

    # Plot trajectories
    mission.duration.plot_trajectory(hover_phase)
    mission.duration.plot_trajectory(cruise_phase)

