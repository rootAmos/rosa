import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from tools.lift_prop_import import EmpiricalPropeller
from tools.emotor_import_rbf import EmpiricalMotor

class HoverPerformance:
    def __init__(self):
        pass

    def analyze(self, vehicle, phase):
            
        """
        Analyze hover performance for 8-motor configuration.
        
        Args:
            vehicle: Vehicle data
            phase: Phase data
        """

        # Unpack vehicle and phase
        mtom_kg = vehicle['mtom_kg']
        density_kgm3 = phase['density_kgm3']
        num_lift_prplsrs = vehicle['num_lift_prplsrs']

        

        # Calculate required thrust per motor (weight / 8)
        weight_N = mtom_kg * 9.806
        phase['ver_thrust_unit_N'] = (weight_N + phase['zddot_m_s2'] * mtom_kg) / num_lift_prplsrs
# Load propeller and motor data
        prop = EmpiricalPropeller(vehicle)
    
        power_lift_motor_W, rpm_lift_motor, coll = prop.calculate_power(phase, vehicle)
        
        
        motor = EmpiricalMotor()    
        interp_func = motor.create_efficiency_interpolator(kernel='thin_plate_spline')
        torque_lift_motor_Nm = power_lift_motor_W / (rpm_lift_motor * 2 * np.pi / 60)
        rpm_torque_points = np.column_stack((rpm_lift_motor, torque_lift_motor_Nm))
        eta_lift_motor = interp_func(rpm_torque_points)

        p_elec_motor_W = power_lift_motor_W / eta_lift_motor * num_lift_prplsrs
        
        # Pack phase
        phase['power_lift_motor_W'] = power_lift_motor_W
        phase['power_elec_W'] = p_elec_motor_W
        phase['rpm_lift_motor'] = rpm_lift_motor
        phase['beta_lift_motor'] = coll
        phase['eta_lift_motor'] = eta_lift_motor
        phase['torque_lift_motor_Nm'] = torque_lift_motor_Nm
        phase['rpm_lift_motor'] = rpm_lift_motor

        return phase

if __name__ == "__main__":

    phase = {}
    phase['N'] = 10
    phase['density_kgm3'] = 1.05 * np.ones(phase['N'])
    phase['udot_m_s2'] = 0 * np.ones(phase['N'])
    phase['zdot_m_s'] = np.zeros(phase['N'])
    phase['zddot_m_s2'] = 0.3
    phase['z0_m'] = 0

    vehicle = {}
    vehicle['mtom_kg'] = 5500  # Example MTOM
    vehicle['lift_prplsr_diam_m'] = 3.09
    vehicle['num_lift_prplsrs'] = 8
    vehicle['lift_prplsr_rpm_min'] = 2200
    vehicle['lift_prplsr_rpm_max'] = 3000
    vehicle['wingarea_m2'] = 23
    vehicle['lift_prplsr_beta_min'] = -7
    vehicle['lift_prplsr_beta_max'] = 21

    hvr = HoverPerformance()
    results = hvr.analyze(vehicle, phase)
    
    print("\nHover Performance Results:")
    print(f"Thrust per motor (N): {results['ver_thrust_unit_N']}")
    print(f"Power per motor (kW): {results['power_lift_motor_W']/1000}")
    print(f"Total Electric Power (kW): {results['power_elec_W']/1000}")
    print(f"Operating RPM: {results['rpm_lift_motor']}")
    print(f"Collective pitch (deg): {results['beta_lift_motor']}")
    print(f"Motor efficiency: {results['eta_lift_motor']}")
    print(f"Torque (Nm): {results['torque_lift_motor_Nm']}")
