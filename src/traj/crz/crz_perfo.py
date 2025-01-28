import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from tools.emotor_import_rbf import EmpiricalMotor
from tools.dfan_import import EmpiricalDfan
from tools.flow5_import import EmpiricalAero

import glob
import pdb

class CruisePerformance:
    def __init__(self, vehicle, phase):
        self.mtom_kg = vehicle['mtom_kg']
        self.density_kg_m3 = phase['density_kgm3']
        self.fwd_prplsr_diam_m = vehicle['fwd_prplsr_diam_m']
        self.u_m_s = phase['u_m_s']
        self.aero_filenames = phase['aero_filenames']
        self.num_lift_prplsrs = vehicle['num_lift_prplsrs']
        self.num_fwd_prplsrs = vehicle['num_fwd_prplsrs']

    def analyze(self):
        """Analyze cruise performance with fully vectorized calculations."""
        # Calculate CL required
        CL = (self.mtom_kg * 9.806) / (0.5 * self.density_kg_m3 * self.u_m_s**2 * np.pi * self.fwd_prplsr_diam_m**2)

        # Create interpolation functions
        aero = EmpiricalAero(self.aero_filenames)
        alpha_from_CL = aero.interp_alpha_from_CL()
        
        # Get alpha and CD using vectorized operations
        speed_CL_points = np.column_stack((self.u_m_s, CL))
        alpha = alpha_from_CL(speed_CL_points)
        
        speed_alpha_points = np.column_stack((self.u_m_s, alpha))
        CD = aero.interp_CD(speed_alpha_points)

        # Vectorized force calculations
        drag_N = 0.5 * self.density_kg_m3 * self.u_m_s**2 * np.pi * self.fwd_prplsr_diam_m**2 * CD
        thrust_N = drag_N
        unit_thrust_N = thrust_N / self.num_fwd_prplsrs

        dfan = EmpiricalDfan()

        # Get propeller and motor performance (assuming dfan.calculate_power is vectorized)
        power_W, rpm = dfan.calculate_power(
            unit_thrust_N, 
            self.u_m_s, 
            self.density_kg_m3, 
            self.fwd_prplsr_diam_m
        )

        # Calculate motor efficiency
        torque_Nm = power_W / (rpm * 2 * np.pi / 60)
        rpm_torque_points = np.column_stack((rpm, torque_Nm))
        eta_motor = self.motor.interp_efficiency(rpm_torque_points)
        
        p_elec_motor_W = power_W / eta_motor

        return {
            'thrust_N': thrust_N,
            'power_W': power_W,
            'power_elec_W': p_elec_motor_W,
            'operating_rpm': rpm,
            'eta_motor': eta_motor,
        }

if __name__ == "__main__":

    # Example usage
    vehicle = {}
    vehicle['mtom_kg'] = 5500  # Example MTOM
    vehicle['fwd_prplsr_diam_m'] = 2.7
    vehicle['num_fwd_prplsrs'] = 4
    vehicle['num_lift_prplsrs'] = 4

    phase = {}
    phase['len'] = 10
    phase['density_kgm3'] = 1.225 * np.ones(phase['len'])
    phase['u_m_s'] = 60 * np.ones(phase['len']) # airspeed in the body axis x-direction
    phase['aero_filenames'] = glob.glob("data/aero/*.txt")
    
    crz = CruisePerformance(vehicle, phase)
    
    results = crz.analyze()
    
    print("\nHover Performance Results:")
    print(f"Thrust per motor: {results['thrust_N'][0]:.1f} N")
    print(f"Power per motor: {results['power_W'][0]/1000:.1f} kW")
    print(f"Power elec per motor: {results['power_elec_W'][0]/1000:.1f} kW")
    print(f"Operating RPM: {results['operating_rpm'][0]:.1f}")
    print(f"Motor efficiency: {results['eta_motor']:.3f}")
