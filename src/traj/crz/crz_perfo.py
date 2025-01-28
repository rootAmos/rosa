import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from tools.emotor_import_rbf import EmpiricalMotor
from tools.dfan_import import EmpiricalDfan
from tools.flow5_import import EmpiricalAero
import matplotlib.pyplot as plt

import glob
import pdb

class CruisePerformance:
    def __init__(self, phase):
        self.aero_filenames = phase['aero_filenames']

    def analyze(self, vehicle, phase):
        """Analyze cruise performance with fully vectorized calculations."""

        # Unpack vehicle and phase
        mtom_kg = vehicle['mtom_kg']
        density_kg_m3 = phase['density_kgm3']
        fwd_prplsr_diam_m = vehicle['fwd_prplsr_diam_m']
        u_m_s = phase['u_m_s']
        num_fwd_prplsrs = vehicle['num_fwd_prplsrs']

        # Calculate CL required
        CL = (mtom_kg * 9.806) / (0.5 * density_kg_m3 * u_m_s**2 * vehicle['wingarea_m2'])

        # Create interpolation functions
        aero = EmpiricalAero(self.aero_filenames)
        alpha_from_CL = aero.interp_alpha_from_CL()
        
        # Get alpha and CD using vectorized operations
        speed_CL_points = np.vstack((u_m_s, CL))
        alpha = alpha_from_CL(speed_CL_points)
        
        speed_alpha_points = np.vstack((u_m_s, alpha))
        CD_from_alpha = aero.interp_coeff_from_alpha('CD')
        CD = CD_from_alpha(speed_alpha_points)

        # Vectorized force calculations
        drag_N = 0.5 * density_kg_m3 * u_m_s**2 * vehicle['wingarea_m2'] * CD
        thrust_N = drag_N + vehicle['mtom_kg'] * phase['udot_m_s2']
        unit_thrust_N = thrust_N / num_fwd_prplsrs


        phase['thrust_N'] = thrust_N
        phase['drag_N'] = drag_N
        phase['lift_N'] = vehicle['mtom_kg'] * 9.806 * np.ones(phase['N'])
        phase['CL'] = CL
        phase['CD'] = CD
        phase['alpha'] = alpha
        phase['thrust_unit_N'] = unit_thrust_N

        dfan = EmpiricalDfan(vehicle)

        # Get propeller and motor performance (assuming dfan.calculate_power is vectorized)
        power_fwd_motor_W, rpm = dfan.calculate_power(vehicle, phase)



        # Calculate motor efficiency
        motor = EmpiricalMotor()
        interp_func = motor.create_efficiency_interpolator(kernel='thin_plate_spline')

        torque_fwd_motor_Nm = power_fwd_motor_W / (rpm * 2 * np.pi / 60)
        rpm_torque_points = np.column_stack((rpm, torque_fwd_motor_Nm))
        eta_fwd_motor = interp_func(rpm_torque_points)
        
        p_elec_motor_W = power_fwd_motor_W / eta_fwd_motor * vehicle['num_fwd_prplsrs']

        # pack phase
        phase['power_fwd_motor_W'] = power_fwd_motor_W
        phase['power_elec_W'] = p_elec_motor_W
        phase['operating_rpm'] = rpm
        phase['eta_fwd_motor'] = eta_fwd_motor
        phase['torque_fwd_motor_Nm'] = torque_fwd_motor_Nm  
        phase['rpm_fwd_motor'] = rpm

        return phase
    
    def plot_performance(self, phase):
        # First figure: Forces
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Horizontal forces
        ax1.plot(phase['time_s'], phase['thrust_N'], label='Thrust (N)')
        ax1.plot(phase['time_s'], phase['drag_N'], label='Drag (N)')
        ax1.set_ylabel('Force (N)')
        ax1.set_xlabel('Time (s)')
        ax1.grid(True)
        ax1.legend()
        
        # Vertical forces
        ax2.plot(phase['time_s'], phase['lift_N'], label='Lift (N)')
        ax2.set_ylabel('Force (N)')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout(h_pad=1.0, pad=1.5)
        
        # Second figure: Powers
        fig2, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(phase['time_s'], phase['power_fwd_motor_W']/1000, label='Forward EMotor Unit Power (kW)')
        ax3.plot(phase['time_s'], phase['power_elec_W']/1000, label='Total Electrical Power (kW)')
        ax3.set_ylabel('Power (kW)')
        ax3.set_xlabel('Time (s)')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        
        # Third figure: Motor characteristics
        fig3, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Motor efficiency
        ax4.plot(phase['time_s'], phase['eta_fwd_motor'], label='Forward EMotor Efficiency')
        ax4.set_ylabel('Efficiency')
        ax4.set_xlabel('Time (s)')
        ax4.grid(True)
        ax4.legend()
        
        # Motor torque
        ax5.plot(phase['time_s'], phase['torque_fwd_motor_Nm'], label='Forward EMotor Torque (Nm)')
        ax5.set_ylabel('Torque (Nm)')
        ax5.set_xlabel('Time (s)')
        ax5.grid(True)
        ax5.legend()
        
        # Motor RPM
        ax6.plot(phase['time_s'], phase['rpm_fwd_motor'], label='Forward EMotor RPM')
        ax6.set_ylabel('RPM')
        ax6.set_xlabel('Time (s)')
        ax6.grid(True)
        ax6.legend()
        
        plt.tight_layout(h_pad=1.0, pad=1.5)
        
        # Fourth figure: Aerodynamic coefficients
        fig4, (ax7, ax8, ax9) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Lift coefficient
        ax7.plot(phase['time_s'], phase['CL'], label='$C_L$')
        ax7.set_ylabel('$C_L$')
        ax7.set_xlabel('Time (s)')
        ax7.grid(True)
        ax7.legend()
        
        # Drag coefficient
        ax8.plot(phase['time_s'], phase['CD'], label='$C_D$')
        ax8.set_ylabel('$C_D$')
        ax8.set_xlabel('Time (s)')
        ax8.grid(True)
        ax8.legend()
        
        # Angle of attack
        ax9.plot(phase['time_s'], phase['alpha'], label='$\\alpha$ (rad)')
        ax9.set_ylabel('$\\alpha$ (rad)')
        ax9.set_xlabel('Time (s)')
        ax9.grid(True)
        ax9.legend()
        
        plt.tight_layout(h_pad=1.0, pad=1.5)
        
        plt.show()

if __name__ == "__main__":

    # Example usage
    vehicle = {}
    vehicle['mtom_kg'] = 5500  # Example MTOM
    vehicle['fwd_prplsr_diam_m'] = 1.58
    vehicle['num_fwd_prplsrs'] = 2
    vehicle['num_lift_prplsrs'] = 8
    vehicle['fwd_prplsr_rpm_min'] = 2200
    vehicle['fwd_prplsr_rpm_max'] = 3000
    vehicle['wingarea_m2'] = 23
    

    phase = {}
    phase['N'] = 10
    phase['density_kgm3'] = 1.05 * np.ones(phase['N'])
    phase['u_m_s'] = 70 * np.ones(phase['N']) # airspeed in the body axis x-direction
    phase['aero_filenames'] = glob.glob("data/aero/*.txt")
    phase['dur_s'] = 2 *60 * 60
    phase['udot_m_s2'] = 0.05 * np.ones(phase['N'])
    phase['dt_s'] = phase['dur_s'] / phase['N']
    phase['time_s'] = np.arange(0, phase['dur_s'], phase['dt_s'])

    
    crz = CruisePerformance(phase)
    
    phase = crz.analyze(vehicle, phase )
    
    print("\nCruise Performance Results:")
    print(f"Thrust per motor: {phase['thrust_N']} N")
    print(f"Power per motor: {phase['power_fwd_motor_W']/1000} kW")
    print(f"Power elec per motor: {phase['power_elec_W']/1000} kW")
    print(f"Operating RPM: {phase['operating_rpm']} RPM")
    print(f"Motor efficiency: {phase['eta_fwd_motor']}")

    crz.plot_performance(phase)
